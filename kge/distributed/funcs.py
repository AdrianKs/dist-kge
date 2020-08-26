import os
from typing import Dict, Optional
from kge import Config, Dataset
from kge.distributed.parameter_server import init_torch_server, init_lapse_scheduler
from kge.distributed.worker_process import WorkerProcessPool
from kge.distributed.work_scheduler import WorkScheduler
from kge.distributed.misc import MIN_RANK

import torch
from torch import multiprocessing as mp


def create_and_run_distributed(config: Config, dataset: Optional[Dataset] = None,
                               checkpoint: Optional[Dict] = None):
    os.environ["OMP_NUM_THREADS"] = str(
        config.get("job.distributed.num_threads_per_process")
    )
    os.environ["GLOO_SOCKET_IFNAME"] = config.get(
        "job.distributed.gloo_socket_ifname")
    processes = []
    num_keys = dataset.num_entities() + dataset.num_relations()
    num_meta_keys = 2
    num_workers = config.get("job.distributed.num_workers")
    master_ip = config.get("job.distributed.master_ip")
    master_port = config.get("job.distributed.master_port")
    lapse_port = config.get("job.distributed.lapse_port")
    num_partitions = config.get("job.distributed.num_partitions")
    dist_world_size = num_workers + MIN_RANK
    dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, dim)
    if config.get("train.optimizer") in ["dist_adagrad", "dist_rowadagrad"]:
    #    num_keys *= 2
        num_meta_keys += 2
    # meta keys. contains for example a variable indicating whether to stop or
    #  not
    num_keys += num_meta_keys
    # todo: we should define server, scheduler and worker ranks here
    #  then create a extra dist worker group after every init
    #  we can create the scheduler-clients in the worker generation
    #  and provide them with the scheduler rank
    #  then we can remove this ugly mock process and can have a barrier
    #  for the workers only
    if config.get("job.distributed.machine_id") == 0:
        if config.get("job.distributed.parameter_server") == "lapse":
            p = mp.Process(
                target=init_lapse_scheduler,
                args=(
                    num_workers,
                    num_keys,
                    master_ip,
                    master_port,
                    lapse_port,
                    dist_world_size,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)
        else:
            p = mp.Process(
                target=init_torch_server,
                args=(num_workers, num_keys, dim + optimizer_dim, master_ip, master_port),
                daemon=True,
            )
            p.start()
            processes.append(p)

        # create a work scheduler
        partition_type = config.get("job.distributed.partition_type")
        scheduler = WorkScheduler.create(
            partition_type=partition_type,
            world_size=num_workers + 2,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_workers,
            dataset=dataset,
            dataset_folder=dataset.folder,
            scheduling_order=config.get("job.distributed.scheduling_order"),
            repartition_epoch=config.get("job.distributed.repartition_epoch"),
        )
        scheduler.start()
        processes.append(scheduler)
    num_workers = config.get("job.distributed.num_workers")
    num_workers_machine = config.get("job.distributed.num_workers_machine")
    if num_workers_machine <= 0:
        num_workers_machine = num_workers
    already_init_workers = config.get("job.distributed.already_init_workers")
    worker_process_pool = WorkerProcessPool(
        num_workers, num_workers_machine, already_init_workers, num_keys,
        num_meta_keys, dim, optimizer_dim, config, dataset, checkpoint
    )
    worker_process_pool.join()
    for p in processes:
        p.join()

def get_optimizer_dim(config: Config, dim):
    optimizer = config.get("train.optimizer")
    if optimizer == "dist_sgd":
        optimizer_dim = 0
    elif optimizer == "dist_adagrad":
        optimizer_dim = dim
    elif optimizer == "dist_rowadagrad":
        optimizer_dim = 1
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
    return optimizer_dim
