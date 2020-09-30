import os
import gc
import torch  # import torch before lapse
import lapse
from typing import Optional, Dict
from copy import deepcopy
from torch import multiprocessing as mp
from torch import distributed as dist

from kge import Dataset
from kge.misc import set_seeds
from kge.job import Job
from .parameter_client import KgeParameterClient
from .misc import MIN_RANK


class WorkerProcessPool:
    """
    Creates all the train-workers for distributed training
    """
    def __init__(
        self,
        num_total_workers,
        num_workers_machine,
        already_init_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        optimizer_dim,
        config,
        dataset,
        checkpoint: Optional[Dict] = None,
    ):
        self.workers = []
        configs = {}
        parameters=None
        if config.get("job.distributed.parameter_server") == "shared":
            parameters = torch.empty((num_keys, embedding_dim + optimizer_dim), dtype=torch.float32, requires_grad=False).share_memory_()
        for rank in range(num_workers_machine):
            if rank == 0:
                self.recv_end, send_end = mp.Pipe(False)
            else:
                send_end = None
            configs[rank] = deepcopy(config)
            configs[rank].set(config.get("model") + ".create_complete", False)
            configs[rank].init_folder()
            worker = WorkerProcess(
                rank + already_init_workers,
                num_total_workers,
                num_keys,
                num_meta_keys,
                embedding_dim,
                optimizer_dim,
                configs[rank],
                dataset,
                parameters=parameters,
                checkpoint=checkpoint,
                result_pipe=send_end
            )
            worker.start()
            self.workers.append(worker)

    def join(self):
        """Wait for all workers"""
        valid_trace = self.recv_end.recv()
        for worker in self.workers:
            worker.join()
        return valid_trace


class WorkerProcess(mp.get_context("spawn").Process):
    """Train worker"""
    def __init__(
        self,
        rank,
        num_total_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        optimizer_dim,
        config,
        dataset,
        parameters=None,
        checkpoint: Optional[Dict] = None,
        result_pipe=None,
    ):
        # rank = rank + 1
        daemon = config.get("train.num_workers") <= 0
        super().__init__(daemon=daemon, name=f"Worker #{rank}")
        self.rank = rank
        self.num_total_workers = num_total_workers
        self.num_keys = num_keys
        self.num_meta_keys = num_meta_keys
        self.embedding_dim = embedding_dim
        self.optimizer_dim = optimizer_dim
        self.config = config
        self.dataset = dataset
        self.parameters = parameters
        self.checkpoint = checkpoint
        self.result_pipe = result_pipe

    def run(self):
        # seeds need to be set in every process
        set_seeds(self.config, self.rank)

        os.environ["MASTER_ADDR"] = self.config.get("job.distributed.master_ip")
        os.environ["MASTER_PORT"] = self.config.get("job.distributed.master_port")
        print("before init", self.rank + MIN_RANK)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self.num_total_workers + MIN_RANK,
            rank=self.rank + MIN_RANK,
        )
        worker_ranks = list(range(MIN_RANK, self.num_total_workers+MIN_RANK))
        worker_group = dist.new_group(worker_ranks)

        # create parameter server
        server = None
        if self.config.get("job.distributed.parameter_server") == "lapse":
            os.environ["DMLC_NUM_WORKER"] = "0"
            os.environ["DMLC_NUM_SERVER"] = str(self.num_total_workers)
            os.environ["DMLC_ROLE"] = "server"
            os.environ["DMLC_PS_ROOT_URI"] = self.config.get(
                "job.distributed.master_ip"
            )
            os.environ["DMLC_PS_ROOT_PORT"] = self.config.get(
                "job.distributed.lapse_port"
            )

            num_workers_per_server = 1
            lapse.setup(self.num_keys, num_workers_per_server)
            server = lapse.Server(self.num_keys, self.embedding_dim + self.optimizer_dim)
        elif self.config.get("job.distributed.parameter_server") == "shared":
            server = self.parameters

        # create train-worker config, dataset and folder
        device_pool: list = self.config.get("job.device_pool")
        if len(device_pool) == 0:
            device_pool.append(self.config.get("job.device"))
        worker_id = self.rank
        config = deepcopy(self.config)
        config.set("job.device", device_pool[worker_id % len(device_pool)])
        config.folder = os.path.join(self.config.folder, f"worker-{self.rank}")
        config.init_folder()
        dataset = deepcopy(self.dataset)

        parameter_client = KgeParameterClient.create(
            client_type=self.config.get("job.distributed.parameter_server"),
            server_id=0,
            client_id=worker_id + MIN_RANK,
            embedding_dim=self.embedding_dim + self.optimizer_dim,
            server=server,
            num_meta_keys=self.num_meta_keys,
            worker_group=worker_group,
        )
        init_for_load_only = False

        # load data from checkpoint and create job
        if parameter_client.rank == MIN_RANK and self.checkpoint is not None:
            # Todo: we still create a complete new job after creating the resume job
            #  therefore epoch numbers will not be handled correctly, for example
            job = Job.create_from(self.checkpoint, parameter_client=parameter_client)
            job.model.get_s_embedder().push_all()
            job.model.get_p_embedder().push_all()
            init_for_load_only = True

        job = Job.create(
            config=config,
            dataset=dataset,
            parameter_client=parameter_client,
            init_for_load_only=init_for_load_only,
        )
        if parameter_client.rank == MIN_RANK and self.checkpoint is not None:
            job.epoch = self.checkpoint["epoch"]
            job.valid_trace = self.checkpoint["valid_trace"]
            del self.checkpoint

        job.run()

        # all done, clean up
        job.work_scheduler_client.shutdown()
        parameter_client.shutdown()
        # delete all occurrences of the parameter client to properly shutdown lapse
        # del job
        del job.parameter_client
        del job.model.get_s_embedder().parameter_client
        del job.model.get_p_embedder().parameter_client
        del job.model
        del job.optimizer
        del parameter_client
        gc.collect()  # make sure lapse-worker destructor is called
        # shutdown server
        if server is not None:
            server.shutdown()
        if self.result_pipe is not None:
            self.result_pipe.send(job.valid_trace)
