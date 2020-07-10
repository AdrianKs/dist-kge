import torch
import lapse
import numpy as np
from typing import Optional
from torch import distributed as dist

PULL_CMD = 0
PUSH_CMD = 1
BARRIER_CMD = 2
SHUTDOWN_CMD = 3


class KgeParameterClient:
    def pull(self, keys, pull_tensor=None, asynchronous=False):
        raise NotImplementedError()

    def push(self, keys, push_tensor, asynchronous=False):
        raise NotImplementedError()

    def localize(self, keys, asynchronous=False):
        raise NotImplementedError()

    def barrier(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()

    def is_stopped(self):
        return False

    @staticmethod
    def create(client_type, server_id, client_id, embedding_dim, server=None, num_meta_keys=0):
        if client_type == "lapse":
            return LapseParameterClient(
                server_id,
                rank=client_id,
                lapse_server=server,  # in lapse we need to provide the actual server
                num_meta_keys=num_meta_keys,
            )
        if client_type == "torch":
            return TorchParameterClient(
                server_rank=server_id, rank=client_id, dim=embedding_dim
            )
        else:
            raise ValueError(client_type)


class LapseParameterClient(lapse.Worker, KgeParameterClient):
    def __init__(
        self,
        customer_id: int,
        rank: int,
        lapse_server: lapse.Server,
        num_meta_keys,
    ):
        super(LapseParameterClient, self).__init__(customer_id, rank, lapse_server)
        self.rank = rank
        self.num_meta_keys = num_meta_keys
        self._stop_key = torch.LongTensor([self.num_keys - self.num_meta_keys])
        self._optim_entity_step_key = torch.LongTensor(
            [self.num_keys - self.num_meta_keys + 1]
        )
        self._optim_relation_step_key = torch.LongTensor(
            [self.num_keys - self.num_meta_keys + 2]
        )
        self._stop_value_tensor = torch.zeros((1, self.key_size), dtype=torch.float32)
        self._optim_entity_step_value_tensor = torch.zeros(
            (1, self.key_size), dtype=torch.float32
        )
        self._optim_relation_step_value_tensor = torch.zeros(
            (1, self.key_size), dtype=torch.float32
        )
        self.meta_key_tensor = torch.zeros(
            (self.num_meta_keys, self.key_size), dtype=torch.float32
        )

    def pull(
        self, keys, pull_tensor: Optional[torch.Tensor] = None, asynchronous=False
    ):
        # if type(keys) is torch.Tensor:
        #     keys = keys.numpy.astype(np.unint64)
        if pull_tensor is None:
            pull_tensor = torch.zeros([len(keys), self.key_size], dtype=torch.float32)
        super(LapseParameterClient, self).pull(keys, pull_tensor, asynchronous)

    def push(self, keys, push_tensor: torch.Tensor, asynchronous=False):
        super(LapseParameterClient, self).push(keys, push_tensor)

    def localize(self, keys, asynchronous=False):
        if type(keys) is torch.Tensor:
            keys = keys.numpy().astype(np.uint64)
        super(LapseParameterClient, self).localize(keys, asynchronous)

    def shutdown(self):
        super(LapseParameterClient, self).push(
            self._stop_key, torch.ones((1, self.key_size), dtype=torch.float32)
        )

    def is_stopped(self) -> bool:
        super(LapseParameterClient, self).pull(self._stop_key, self._stop_value_tensor)
        if torch.any(self._stop_value_tensor[0] == 1):
            return True
        else:
            return False

    def step_optim(self, parameter_index):
        if parameter_index == 0:
            super(LapseParameterClient, self).push(
                self._optim_entity_step_key,
                torch.ones((1, self.key_size), dtype=torch.float32),
            )
        else:
            super(LapseParameterClient, self).push(
                self._optim_relation_step_key,
                torch.ones((1, self.key_size), dtype=torch.float32),
            )

    def get_step_optim(self, parameter_index):
        if parameter_index == 0:
            super(LapseParameterClient, self).pull(
                self._optim_entity_step_key, self._optim_entity_step_value_tensor
            )
            return self._optim_relation_step_value_tensor[0, 0].item()
        else:
            super(LapseParameterClient, self).pull(
                self._optim_relation_step_key, self._optim_relation_step_value_tensor
            )
            return self._optim_relation_step_value_tensor[0, 0].item()


class TorchParameterClient(KgeParameterClient):
    def __init__(self, server_rank, rank, dim):
        self.server_rank = server_rank
        self.rank = rank
        self.dim = dim
        self.data_type = torch.float32

    def pull(self, keys, pull_tensor=None, asynchronous=False):
        cmd = torch.LongTensor([PULL_CMD, len(keys)])
        dist.send(cmd, dst=self.server_rank)
        dist.send(keys, dst=self.server_rank)
        if pull_tensor is None:
            pull_tensor = torch.zeros((len(keys), self.dim), dtype=self.data_type)
        dist.recv(pull_tensor, src=self.server_rank)

    def push(self, keys, push_tensor, asynchronous=False):
        cmd = torch.LongTensor([PUSH_CMD, len(keys)])
        dist.send(cmd, dst=self.server_rank)
        dist.send(keys, dst=self.server_rank)
        dist.send(push_tensor, dst=self.server_rank)

    def localize(self, keys, asynchronous=False):
        pass

    def barrier(self):
        cmd = torch.LongTensor([BARRIER_CMD, 0])
        dist.send(cmd, dst=self.server_rank)
        dist.barrier()

    def shutdown(self):
        cmd = torch.LongTensor([SHUTDOWN_CMD, 0])
        dist.send(cmd, dst=self.server_rank)

    def step_optim(self, parameter_index):
        # todo we still need to implement the handling of optimizer steps
        pass

    def get_step_optim(self, parameter_index):
        # todo we still need to implement the handling of optimizer steps
        return 1
