import os
import time
import logging
import psutil
from typing import Dict, Optional
from kge import Config, Dataset
from kge.distributed.parameter_server import init_torch_server, init_lapse_scheduler
from kge.distributed.worker_process import WorkerProcessPool
from kge.distributed.work_scheduler import WorkScheduler
from kge.distributed.misc import get_optimizer_dim, get_min_rank

import torch
from torch import multiprocessing as mp


def monitor_hardware(folder, interval=1):
    def bytes_to_mb(bytes_amount):
        return round(bytes_amount / 1024 / 1024, 2)
    logger = logging.getLogger('hardware_monitor')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(folder, 'hardware_monitor.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    while True:
        time.sleep(interval)
        cpu_percentage = psutil.cpu_percent()
        memory_percentage = psutil.virtual_memory().percent
        network_info = psutil.net_io_counters()
        logger.info(msg=f"{cpu_percentage};{memory_percentage};{bytes_to_mb(network_info.bytes_sent)};{bytes_to_mb(network_info.bytes_recv)}")


def create_and_run_distributed(
    config: Config, dataset: Optional[Dataset] = None, checkpoint: Optional[Dict] = None
):
    os.environ["OMP_NUM_THREADS"] = str(
        config.get("job.distributed.num_threads_per_process")
    )
    os.environ["GLOO_SOCKET_IFNAME"] = config.get("job.distributed.gloo_socket_ifname")
    processes = []
    num_keys = dataset.num_entities() + dataset.num_relations()
    num_meta_keys = 3
    num_workers = config.get("job.distributed.num_workers")
    master_ip = config.get("job.distributed.master_ip")
    master_port = config.get("job.distributed.master_port")
    lapse_port = config.get("job.distributed.lapse_port")
    num_partitions = config.get("job.distributed.num_partitions")
    min_rank = get_min_rank(config)
    dist_world_size = num_workers + min_rank
    dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, dim)
    if config.get("train.optimizer.default.type") in [
        "dist_adagrad",
        "dist_rowadagrad",
    ]:
        #    num_keys *= 2
        num_meta_keys += 2
    # meta keys. contains for example a variable indicating whether to stop or
    #  not
    num_keys += num_meta_keys

    if config.get("job.distributed.repartition_epoch") and config.get("job.distributed.partition_type") == "2d_block_partition":
        # with stratificaton we have a lot of open files that need to be shared
        # between processes. Some servers don't allow that. Therefore set sharing
        # strategy to file_system to avoid too many open files error
        torch.multiprocessing.set_sharing_strategy('file_system')

    # start hardware monitoring
    monitor_process = mp.Process(
        target=monitor_hardware, args=(config.folder, 0.5), daemon=True
    )
    monitor_process.start()

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
                    min_rank,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)
        elif config.get("job.distributed.parameter_server") == "torch":
            p = mp.Process(
                target=init_torch_server,
                args=(
                    num_workers,
                    num_keys,
                    dim + optimizer_dim,
                    master_ip,
                    master_port,
                    min_rank,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        # create a work scheduler
        partition_type = config.get("job.distributed.partition_type")
        print("init scheduler")
        scheduler_init_time = time.time()
        scheduler = WorkScheduler.create(
            config=config,
            partition_type=partition_type,
            world_size=num_workers + min_rank,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_workers,
            dataset=dataset,
            dataset_folder=dataset.folder,
            scheduling_order=config.get("job.distributed.scheduling_order"),
            repartition_epoch=config.get("job.distributed.repartition_epoch"),
        )
        config.log(f"scheduler initialized after: {time.time()-scheduler_init_time}")
        print("start scheduler")
        scheduler_start_time = time.time()
        scheduler.start()
        processes.append(scheduler)
        config.log(f"scheduler start took: {time.time()-scheduler_start_time}")

    # create all train-workers in a worker pool
    num_workers = config.get("job.distributed.num_workers")
    num_workers_machine = config.get("job.distributed.num_workers_machine")
    if num_workers_machine <= 0:
        num_workers_machine = num_workers
    already_init_workers = config.get("job.distributed.already_init_workers")
    worker_process_pool = WorkerProcessPool(
        num_workers,
        num_workers_machine,
        already_init_workers,
        num_keys,
        num_meta_keys,
        dim,
        optimizer_dim,
        config,
        dataset,
        checkpoint,
    )
    valid_trace = worker_process_pool.join()
    for p in processes:
        p.join()
    monitor_process.terminate()
    return valid_trace
