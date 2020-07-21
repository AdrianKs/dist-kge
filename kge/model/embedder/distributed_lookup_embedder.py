from torch import Tensor
import torch.nn
import torch.nn.functional
import numpy as np

import torch
import lapse

from kge import Config, Dataset
from kge.model import LookupEmbedder, KgeEmbedder
#from kge.distributed import KgeParameterClient

from typing import List


class DistributedLookupEmbedder(LookupEmbedder):
    def __init__(
        self, config: Config, dataset: Dataset, configuration_key: str, vocab_size: int, parameter_client: "KgeParameterClient", lapse_index: torch.Tensor, complete_vocab_size, init_for_load_only=False
    ):
        super().__init__(config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)

        self.complete_vocab_size = complete_vocab_size
        self.parameter_client = parameter_client
        self.lapse_index = lapse_index  # maps the id from the dataset to the id stored in lapse
        self.local_index_mapper = torch.zeros(self.complete_vocab_size, dtype=torch.long)-1  # maps the id from the dataset to the id of the embedding here in the embedder
        self.local_to_lapse_mapper = torch.zeros(vocab_size, dtype=torch.long)-1  # maps the local embeddings to the embeddings in lapse
        self.num_pulled = 0

    def push_all(self):
        self.parameter_client.push(self.lapse_index[torch.arange(self.vocab_size)],
                                   self._embeddings.weight.detach().cpu())

    def pull_all(self):
        self._pull_embeddings(torch.arange(self.complete_vocab_size))

    @torch.no_grad()
    def _pull_embeddings(self, indexes):
        local_indexes = self.local_index_mapper[indexes]
        missing_mask = local_indexes == -1
        num_missing = torch.sum(missing_mask).item()
        if num_missing > 0:
            missing_local_indexes = torch.arange(self.num_pulled, self.num_pulled+num_missing, dtype=torch.long)
            self.num_pulled += num_missing

            #self.lapse_worker.localize(keys=self.lapse_index[missing_local_indexes])
            pull_indexes = self.lapse_index[indexes[missing_mask].cpu()].reshape(-1)
            current_embeddings = self._embeddings.weight[missing_local_indexes, :].detach().cpu()
            self.parameter_client.pull(pull_indexes,
                                       current_embeddings)
            self._embeddings.weight[missing_local_indexes, :] = current_embeddings.to(self._embeddings.weight.device)

            # update local index mapper
            self.local_index_mapper[indexes[missing_mask]] = missing_local_indexes
            # update local to lapse mapper
            self.local_to_lapse_mapper[missing_local_indexes.numpy()] = pull_indexes

    def localize(self, indexes: Tensor):
        unique_indexes = torch.unique(indexes).cpu()
        self.parameter_client.localize(unique_indexes)
        # TODO: also pull the embeddings and store in a tensor on gpu
        #  this needs to be handled in the background somehow
        #  to device can be done in background, but this needs to wait for localize

    def _embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        with torch.no_grad():
            long_unique_indexes = torch.unique(long_indexes)
            self._pull_embeddings(long_unique_indexes)
        return self._embeddings(self.local_index_mapper[long_indexes].to(self._embeddings.weight.device).long())

    def embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        with torch.no_grad():
            long_unique_indexes = torch.unique(long_indexes)
            self._pull_embeddings(long_unique_indexes)
        return self._postprocess(self._embeddings(self.local_index_mapper[long_indexes].to(self._embeddings.weight.device).long()))

    def embed_all(self) -> Tensor:
        raise NotImplementedError

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

    def _embeddings_all(self) -> Tensor:
        # TODO: this should not be possible in the distributed lookup embedder
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        # Avoid calling lookup embedder penalty and instead call KgeEmbedder penalty
        result = KgeEmbedder.penalty(self, **kwargs)
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
                parameters = self._embed(unique_indexes)
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
