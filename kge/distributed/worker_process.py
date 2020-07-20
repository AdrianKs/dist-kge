import os
import gc
import torch  # import torch before lapse
import lapse
from typing import Optional, Dict
from copy import deepcopy
from torch import multiprocessing as mp
from torch import distributed as dist

from kge import Dataset
from kge.job import Job
from .parameter_client import KgeParameterClient


class WorkerProcessPool:
    def __init__(
        self,
        num_total_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        config,
        dataset,
        checkpoint: Optional[Dict] = None,
    ):
        self.workers = []
        configs = {}
        for rank in range(num_total_workers):
            configs[rank] = deepcopy(config)
            configs[rank].set(config.get("model") + ".create_complete", False)
            configs[rank].folder = os.path.join(config.folder, f"server-{rank}")
            configs[rank].init_folder()
            worker = WorkerProcess(
                rank,
                num_total_workers,
                num_keys,
                num_meta_keys,
                embedding_dim,
                configs[rank],
                dataset,
                checkpoint,
            )
            worker.start()
            self.workers.append(worker)

    def join(self):
        for worker in self.workers:
            worker.join()


class WorkerProcess(mp.get_context("spawn").Process):
    def __init__(
        self,
        rank,
        num_total_workers,
        num_keys,
        num_meta_keys,
        embedding_dim,
        config,
        dataset,
        checkpoint: Optional[Dict] = None,
    ):
        # rank = rank + 1
        super().__init__(daemon=True, name=f"Worker #{rank}")
        self.rank = rank
        self.num_total_workers = num_total_workers
        self.num_keys = num_keys
        self.num_meta_keys = num_meta_keys
        self.embedding_dim = embedding_dim
        self.config = config
        self.dataset = dataset
        self.checkpoint = checkpoint

    def run(self):
        os.environ["MASTER_ADDR"] = self.config.get("job.distributed.master_ip")
        os.environ["MASTER_PORT"] = self.config.get("job.distributed.master_port")
        print("before init", self.rank + 2)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self.num_total_workers + 2,
            rank=self.rank + 2,
        )
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
            server = lapse.Server(self.num_keys, self.embedding_dim)

        configs = {}
        datasets = {}
        w = 0
        device_pool: list = self.config.get("job.device_pool")
        if len(device_pool) == 0:
            device_pool.append(self.config.get("job.device"))
        worker_id = self.rank
        configs[w] = deepcopy(self.config)
        configs[w].set("job.device", device_pool[worker_id % len(device_pool)])
        configs[w].folder = os.path.join(self.config.folder, f"worker-{w}")
        configs[w].init_folder()
        datasets[w] = deepcopy(self.dataset)
        # datasets[w] = Dataset.create(
        #     configs[w],
        #     folder=os.path.join(self.dataset.folder, f"partition_{worker_id}"),
        # )
        # datasets[w] = Dataset.create(configs[w], dataset.folder)
        # kv = lapse.Worker(0, worker_id + 1, s)
        # kv = LapseWorker(0, worker_id + 1, s, num_meta_keys)
        parameter_client = KgeParameterClient.create(
            client_type=self.config.get("job.distributed.parameter_server"),
            server_id=0,
            client_id=worker_id + 2,
            embedding_dim=self.embedding_dim,
            server=server,
            num_meta_keys=self.num_meta_keys,
        )
        init_for_load_only = False
        if parameter_client.rank == 2 and self.checkpoint is not None:
            # Todo: we still create a complete new job after creating the resume job
            #  therefore epoch numbers will not be handled correctly, for example
            job = Job.create_from(self.checkpoint)
            job.model.get_s_embedder().push_all()
            job.model.get_p_embedder().push_all()
            job.optimizer.push_all()
            init_for_load_only = True
            del self.checkpoint
        job = Job.create(
            configs[w],
            datasets[w],
            parameter_client=parameter_client,
            init_for_load_only=init_for_load_only,
        )
        job.run()

        job.work_scheduler_client.shutdown()
        parameter_client.shutdown()
        del job
        del parameter_client
        gc.collect()  # make sure lapse-worker destructor is called
        # shutdown server
        if server is not None:
            server.shutdown()