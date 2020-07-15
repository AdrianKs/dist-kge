import os
import torch
import lapse
from torch import distributed as dist

PULL_CMD = 0
PUSH_CMD = 1
BARRIER_CMD = 2
SHUTDOWN_CMD = 3


class KgeParameterServer:
    @staticmethod
    def get_parameter_server():
        raise NotImplementedError()


class LapseParameterServer:
    @staticmethod
    def get_parameter_server(num_keys, key_size, num_servers, ip, port):
        """In Lapse we have a server for every worker, therefore we don't use a lock"""
        os.environ["DMLC_NUM_WORKER"] = "0"
        os.environ["DMLC_NUM_SERVER"] = str(num_servers)
        os.environ["DMLC_ROLE"] = "server"
        os.environ["DMLC_PS_ROOT_URI"] = ip
        os.environ["DMLC_PS_ROOT_PORT"] = port
        num_workers_per_server = 1
        lapse.setup(num_keys, num_workers_per_server)
        return lapse.Server(num_keys, key_size)


class TorchParameterServer:
    def __init__(self, world_size: int, num_keys: int, dim: int):
        self.rank = 0
        self.num_clients = world_size - 1
        self.dim = dim
        self.data_type = torch.float32
        self.data = torch.zeros((num_keys, dim), dtype=self.data_type)
        self.start()

    def start(self):
        barrier_count = 0
        shutdown_count = 0
        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=torch.long)
            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            key_len = cmd_buffer[1].item()
            if cmd == PULL_CMD:
                keys = self._receive_keys(rank, key_len)
                data = self.data[keys, :]
                dist.send(data, dst=rank)
            if cmd == PUSH_CMD:
                keys = self._receive_keys(rank, key_len)
                self._handle_push(rank, keys)
            if cmd == BARRIER_CMD:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            if cmd == SHUTDOWN_CMD:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down parameter server")
                    break

    @staticmethod
    def _receive_keys(rank, key_len):
        keys = torch.zeros((key_len,), dtype=torch.long)
        dist.recv(keys, src=rank)
        return keys

    def _handle_push(self, rank, keys):
        push_data = torch.zeros((len(keys), self.dim), dtype=self.data_type)
        dist.recv(push_data, src=rank)
        self.data[keys, :] += push_data
