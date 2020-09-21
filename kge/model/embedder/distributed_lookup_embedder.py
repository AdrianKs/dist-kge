import time
from torch import Tensor
import torch.nn
import torch.nn.functional

import torch
from collections import deque

from kge import Config, Dataset
from kge.model import LookupEmbedder, KgeEmbedder
from kge.distributed.misc import get_optimizer_dim

from typing import List


class DistributedLookupEmbedder(LookupEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        parameter_client: "KgeParameterClient",
        complete_vocab_size,
        lapse_offset=0,
        init_for_load_only=False,
    ):
        super().__init__(
            config,
            dataset,
            configuration_key,
            vocab_size,
            init_for_load_only=init_for_load_only,
        )
        self.optimizer_dim = get_optimizer_dim(config, self.dim)
        self.optimizer_values = torch.zeros(
            (self.vocab_size, self.optimizer_dim), dtype=torch.float32
        )

        self.complete_vocab_size = complete_vocab_size
        self.parameter_client = parameter_client
        self.lapse_offset = lapse_offset
        self.pulled_ids = None
        self.load_batch = self.config.get("job.distributed.load_batch")
        # global to local mapper only used in sync level partition
        self.global_to_local_mapper = torch.full(
            (self.dataset.num_entities(),), -1, dtype=torch.long, device="cpu"
        )

        # maps the local embeddings to the embeddings in lapse
        # used in optimizer
        self.local_to_lapse_mapper = torch.full((vocab_size,), -1, dtype=torch.long)
        self.pull_dim = self.dim + self.optimizer_dim

        # 3 pull tensors to pre-pull up to 3 batches
        # first boolean denotes if the tensor is free
        self.pull_tensors = [
            [
                True,
                torch.empty(
                    (self.vocab_size, self.dim + self.optimizer_dim),
                    dtype=torch.float32,
                    device="cpu",
                    requires_grad=False,
                ),
            ],
            [
                True,
                torch.empty(
                    (self.vocab_size, self.dim + self.optimizer_dim),
                    dtype=torch.float32,
                    device="cpu",
                    requires_grad=False,
                ),
            ],
            [
                True,
                torch.empty(
                    (self.vocab_size, self.dim + self.optimizer_dim),
                    dtype=torch.float32,
                    device="cpu",
                    requires_grad=False,
                ),
            ],
        ]
        if "cuda" in config.get("job.device"):
            # only pin tensors if we are using gpu
            # otherwise gpu memory will be allocated for no reason
            for i in range(len(self.pull_tensors)):
                self.pull_tensors[i][1] = self.pull_tensors[i][1].pin_memory()

        self.num_pulled = 0
        self.mapping_time = 0.0
        # self.pre_pulled = None
        self.pre_pulled = deque()

    def to_device(self, move_optim_data=True):
        """Needs to be called after model.to(self.device)"""
        if move_optim_data:
            self.optimizer_values = self.optimizer_values.to(
                self._embeddings.weight.device
            )

    def push_all(self):
        self.parameter_client.push(
            torch.arange(self.vocab_size) + self.lapse_offset,
            torch.cat(
                (self._embeddings.weight.detach().cpu(), self.optimizer_values.cpu()),
                dim=1,
            ),
        )

    def pull_all(self):
        self._pull_embeddings(torch.arange(self.complete_vocab_size))

    def set_embeddings(self):
        lapse_indexes = self.pulled_ids + self.lapse_offset
        num_pulled = len(lapse_indexes)
        set_tensor = torch.cat(
            (
                self._embeddings.weight[:num_pulled].detach(),
                self.optimizer_values[:num_pulled],
            ),
            dim=1,
        ).cpu()
        self.parameter_client.set(lapse_indexes, set_tensor)

    def _get_free_pull_tensor(self):
        for i, (free, pull_tensor) in enumerate(self.pull_tensors):
            if free:
                self.pull_tensors[i][0] = False
                return i, pull_tensor

    @torch.no_grad()
    def pre_pull(self, indexes):
        pull_indexes = (indexes + self.lapse_offset).cpu()
        pull_tensor_index, pull_tensor = self._get_free_pull_tensor()
        pull_tensor = pull_tensor[: len(indexes)]
        pull_future = self.parameter_client.pull(
            pull_indexes, pull_tensor, asynchronous=True
        )
        self.pre_pulled.append(
            {
                "indexes": indexes,
                "pull_indexes": pull_indexes,
                "pull_tensor": pull_tensor,
                "pull_future": pull_future,
                "pull_tensor_index": pull_tensor_index,
            }
        )

    def pre_pulled_to_device(self):
        if len(self.pre_pulled) > 2:
            # id 0 is from the batch currently processed
            # last one is the one pulled from ps
            # we are moving the second last
            self.parameter_client.wait(self.pre_pulled[-2]["pull_future"])
            self.pre_pulled[-2]["pull_tensor"] = self.pre_pulled[-2]["pull_tensor"].to(
                self._embeddings.weight.device, non_blocking=True
            )

    @torch.no_grad()
    def _pull_embeddings(self, indexes):
        cpu_gpu_time = 0.0
        pull_time = 0.0
        device = self._embeddings.weight.device
        len_indexes = len(indexes)
        if len(self.pre_pulled) > 0:
            pre_pulled = self.pre_pulled.popleft()
            self.pulled_ids = pre_pulled["indexes"]
            self.parameter_client.wait(pre_pulled["pull_future"])
            self.local_to_lapse_mapper[:len_indexes] = pre_pulled["pull_indexes"]
            cpu_gpu_time -= time.time()
            pre_pulled_tensor = pre_pulled["pull_tensor"].to(device)
            cpu_gpu_time += time.time()
            pulled_embeddings, pulled_optim_values = torch.split(
                pre_pulled_tensor, [self.dim, self.optimizer_dim], dim=1
            )
            self._embeddings.weight[:len_indexes] = pulled_embeddings
            self.optimizer_values[:len_indexes] = pulled_optim_values
            self.pull_tensors[pre_pulled["pull_tensor_index"]][0] = True
            return pull_time, cpu_gpu_time

        self.pulled_ids = indexes
        pull_indexes = (indexes + self.lapse_offset).cpu()
        self.local_to_lapse_mapper[:len_indexes] = pull_indexes
        pull_tensor = self.pull_tensors[0][1][:len_indexes]
        pull_time -= time.time()
        self.parameter_client.pull(pull_indexes, pull_tensor)
        pull_time += time.time()
        cpu_gpu_time -= time.time()
        pull_tensor = pull_tensor.to(device)
        cpu_gpu_time += time.time()
        pulled_embeddings, pulled_optim_values = torch.split(
            pull_tensor, [self.dim, self.optimizer_dim], dim=1
        )
        self._embeddings.weight[:len_indexes] = pulled_embeddings
        self.optimizer_values[:len_indexes] = pulled_optim_values
        return pull_time, cpu_gpu_time

    def localize(self, indexes: Tensor, make_unique=False):
        if make_unique:
            indexes = torch.unique(indexes)
        self.parameter_client.localize((indexes + self.lapse_offset).cpu())
        # TODO: also pull the embeddings and store in a tensor on gpu
        #  this needs to be handled in the background somehow
        #  to device can be done in background, but this needs to wait for localize

    def _embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._embeddings(long_indexes)

    def embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._postprocess(self._embeddings(long_indexes))

    def embed_all(self) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def push_back(self):
        self.local_to_lapse_mapper[:] = -1
        self.num_pulled = 0

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
