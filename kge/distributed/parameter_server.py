import os
import torch
import lapse
from enum import IntEnum
from torch import distributed as dist

class TORCH_PARAMETER_SERVER_CMDS(IntEnum):
    PULL_CMD = 0
    PUSH_CMD = 1
    SET_CMD = 2
    GET_LR_CMD = 3
    SET_LR_CMD = 4
    GET_OPTIM_STEP_CMD = 5
    STEP_OPTIM_CMD = 6
    BARRIER_CMD = 7
    SHUTDOWN_CMD = 8


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
        self.num_clients = world_size - 2
        self.dim = dim
        self.data_type = torch.float32
        self.data = torch.zeros((num_keys, dim), dtype=self.data_type)
        self.lr = 0
        self.entity_optim_step = 0
        self.relation_optim_step = 0
        self.start()

    def start(self):
        barrier_count = 0
        shutdown_count = 0
        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=torch.long)
            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.PULL_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                data = self.data[keys, :]
                dist.send(data, dst=rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.PUSH_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                self._handle_push(rank, keys)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SET_CMD:
                key_len = cmd_buffer[1].item()
                keys = self._receive_keys(rank, key_len)
                self._handle_set(rank, keys)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.GET_LR_CMD:
                cmd_buffer[1] = self.lr
                dist.send(cmd_buffer, rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SET_LR_CMD:
                lr = cmd_buffer[1]
                self.lr = lr
            if cmd == TORCH_PARAMETER_SERVER_CMDS.GET_OPTIM_STEP_CMD:
                parameter_index = cmd_buffer[1].item()
                if parameter_index == 0:
                    cmd_buffer[1] = self.entity_optim_step
                elif parameter_index == 1:
                    cmd_buffer[1] = self.relation_optim_step
                dist.send(cmd_buffer, rank)
            if cmd == TORCH_PARAMETER_SERVER_CMDS.STEP_OPTIM_CMD:
                parameter_index = cmd_buffer[1].item()
                if parameter_index == 0:
                    self.entity_optim_step += 1
                elif parameter_index == 1:
                    self.relation_optim_step += 1
            if cmd == TORCH_PARAMETER_SERVER_CMDS.BARRIER_CMD:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            if cmd == TORCH_PARAMETER_SERVER_CMDS.SHUTDOWN_CMD:
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

    def _handle_set(self, rank, keys):
        set_data = torch.zeros((len(keys), self.dim), dtype=self.data_type)
        dist.recv(set_data, src=rank)
        self.data[keys, :] = set_data
