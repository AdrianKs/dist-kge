import lapse
import numpy as np
import torch
from typing import Optional


class KgeParameterClient:
    def pull(self, keys, pull_tensor=None, asynchronous=False):
        raise NotImplementedError()

    def push(self, keys, push_tensor, asynchronous=False):
        raise NotImplementedError()

    def localize(self, keys, asynchronous=False):
        raise NotImplementedError()

    def barrier(self):
        raise NotImplementedError()

    @staticmethod
    def create(client_type, customer_id, client_id, server, num_meta_keys=0):
        if client_type == "lapse":
            return LapseParameterClient(
                customer_id,
                worker_id=client_id,
                lapse_server=server,
                num_meta_keys=num_meta_keys,
            )
        else:
            raise ValueError(client_type)


class LapseParameterClient(lapse.Worker, KgeParameterClient):
    def __init__(
        self,
        customer_id: int,
        worker_id: int,
        lapse_server: lapse.Server,
        num_meta_keys,
    ):
        super(LapseParameterClient, self).__init__(customer_id, worker_id, lapse_server)
        self.worker_id = worker_id
        self.num_meta_keys = num_meta_keys
        self._stop_key = np.array([self.num_keys - self.num_meta_keys], dtype=np.uint64)
        self._optim_entity_step_key = np.array(
            [self.num_keys - self.num_meta_keys + 1], dtype=np.uint64
        )
        self._optim_relation_step_key = np.array(
            [self.num_keys - self.num_meta_keys + 2], dtype=np.uint64
        )
        self._stop_value_tensor = np.zeros((1, self.key_size), dtype=np.float32)
        self._optim_entity_step_value_tensor = np.zeros(
            (1, self.key_size), dtype=np.float32
        )
        self._optim_relation_step_value_tensor = np.zeros(
            (1, self.key_size), dtype=np.float32
        )
        self.meta_key_tensor = np.zeros(
            (self.num_meta_keys, self.key_size), dtype=np.float32
        )

    def pull(
        self, keys, pull_tensor: Optional[torch.Tensor] = None, asynchronous=False
    ):
        if type(keys) is torch.Tensor:
            keys = keys.numpy.astype(np.unint64)
        if pull_tensor is None:
            pull_tensor = np.zeros([len(keys), self.key_size], dtype=np.float32)
        elif type(pull_tensor) is torch.Tensor:
            pull_tensor = pull_tensor.numpy()
        super(LapseParameterClient, self).pull(keys, pull_tensor, asynchronous)

    def push(self, keys, push_tensor: torch.Tensor, asynchronous=False):
        if type(keys) is torch.Tensor:
            keys = keys.numpy.astype(np.unint64)
        if type(push_tensor) is torch.Tensor:
            push_tensor = push_tensor.numpy()
        super(LapseParameterClient, self).push(keys, push_tensor)

    def localize(self, keys, asynchronous=False):
        if type(keys) is torch.Tensor:
            keys = keys.numpy().astype(np.uint64)
        super(LapseParameterClient, self).localize(keys, asynchronous)

    def stop(self):
        super(LapseParameterClient, self).push(
            self._stop_key, np.ones((1, self.key_size), dtype=np.float32)
        )

    def is_stopped(self) -> bool:
        super(LapseParameterClient, self).pull(self._stop_key, self._stop_value_tensor)
        if np.any(self._stop_value_tensor[0] == 1):
            return True
        else:
            return False

    def step_optim(self, parameter_index):
        if parameter_index == 0:
            super(LapseParameterClient, self).push(
                self._optim_entity_step_key,
                np.ones((1, self.key_size), dtype=np.float32),
            )
        else:
            super(LapseParameterClient, self).push(
                self._optim_relation_step_key,
                np.ones((1, self.key_size), dtype=np.float32),
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
