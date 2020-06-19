from torch import Tensor
import torch.nn
import torch.nn.functional
import numpy as np

import lapse

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


class DistributedLookupEmbedder(KgeEmbedder):
    def __init__(
        self, config: Config, dataset: Dataset, configuration_key: str, vocab_size: int, lapse_worker: lapse.Worker, lapse_index: np.ndarray, complete_vocab_size
    ):
        super().__init__(config, dataset, configuration_key)

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.normalize_with_grad = self.get_option("normalize.with_grad")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size
        self.complete_vocab_size = complete_vocab_size
        self.cached_indexes = []

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        # setup embedder
        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse
        )

        # initialize weights
        init_ = self.get_option("initialize")
        try:
            init_args = self.get_option("initialize_args." + init_)
        except KeyError:
            init_args = self.get_option("initialize_args")

        # Automatically set arg a (lower bound) for uniform_ if not given
        if init_ == "uniform_" and "a" not in init_args:
            init_args["a"] = init_args["b"] * -1
            self.set_option("initialize_args.a", init_args["a"], log=True)

        self.initialize(self._embeddings.weight.data, init_, init_args)

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)
        self.lapse_worker = lapse_worker
        self.lapse_index = lapse_index  # maps the id from the dataset to the id stored in lapse
        #self.local_index_mapper = torch.arange(complete_vocab_size, dtype=torch.int)
        self.local_index_mapper = torch.zeros(complete_vocab_size, dtype=torch.int)-1  # maps the id from the dataset to the id of the embedding here in the embedder
        self.local_to_lapse_mapper = np.zeros(vocab_size, dtype=np.int)-1  # maps the local embeddings to the embeddings in lapse
        self.num_pulled = 0

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0:

            def normalize_embeddings(job):
                if self.normalize_with_grad:
                    self._embeddings.weight = torch.nn.functional.normalize(
                        self._embeddings.weight, p=self.normalize_p, dim=-1
                    )
                else:
                    with torch.no_grad():
                        self._embeddings.weight = torch.nn.Parameter(
                            torch.nn.functional.normalize(
                                self._embeddings.weight, p=self.normalize_p, dim=-1
                            )
                        )

            job.pre_batch_hooks.append(normalize_embeddings)

    def push_all(self):
        self.lapse_worker.push(self.lapse_index[np.arange(self.vocab_size)],
                               self._embeddings.weight.detach().cpu().numpy())

    def pull_all(self):
        self._pull_embeddings(torch.arange(self.complete_vocab_size))

    def _pull_embeddings(self, indexes):
        local_indexes = self.local_index_mapper[indexes]
        missing_mask = local_indexes == -1
        num_missing = torch.sum(missing_mask).item()
        if num_missing > 0:
            missing_local_indexes = torch.arange(self.num_pulled, self.num_pulled+num_missing, dtype=torch.long)
            self.num_pulled += num_missing

            #self.lapse_worker.localize(keys=self.lapse_index[missing_local_indexes])
            # TODO: we still create a new tensor here. We can not just convert a tensor
            #  to numpy, which needs grad, since grad would have to be dropped
            current_embeddings = self._embeddings.weight[missing_local_indexes, :].detach().cpu().numpy()
            pull_indexes = self.lapse_index[indexes[missing_mask].cpu()].reshape(-1)
            self.lapse_worker.pull(pull_indexes,
                                   current_embeddings)
            self._embeddings.weight[missing_local_indexes, :] = torch.from_numpy(
                current_embeddings).to(self._embeddings.weight.device)

            # update local index mapper
            self.local_index_mapper[indexes[missing_mask]] = missing_local_indexes.int()
            # update local to lapse mapper
            self.local_to_lapse_mapper[missing_local_indexes] = pull_indexes

    def embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        with torch.no_grad():
            long_unique_indexes = torch.unique(long_indexes)
            self._pull_embeddings(long_unique_indexes)
        return self._postprocess(self._embeddings(self.local_index_mapper[long_indexes].to(self._embeddings.weight.device).long()))

    def embed_all(self) -> Tensor:
        raise NotImplementedError
        all_indexes = np.arange(self.vocab_size)
        with torch.no_grad():
            self.lapse_worker.localize(keys=all_indexes)
            self.lapse_worker.pull(all_indexes,
                                   self._embeddings.weight[all_indexes,
                                   :].numpy())
            self.cached_indexes = None

        return self._postprocess(self._embeddings_all())

    @torch.no_grad()
    def push_back(self):
        self.local_index_mapper[:] = -1
        self.local_to_lapse_mapper[:] = -1
        self.num_pulled = 0
        # return
        # if self.cached_indexes is None:
        #     indexes = np.arange(self.vocab_size)
        # else:
        #     indexes = np.unique(np.concatenate(self.cached_indexes))
        # self.lapse_worker.push(self.lapse_index[indexes], self._embeddings.weight[indexes, :].numpy())
        # self.cached_indexes = []

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        return self._embeddings(
            torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )
        )

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embeddings(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
