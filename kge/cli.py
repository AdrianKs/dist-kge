#!/usr/bin/env python
import datetime
import os
import sys
import traceback
import yaml
import time

import torch
import lapse

from kge import Dataset
from kge import Config
from kge.job import Job
from kge.misc import get_git_revision_short_hash, kge_base_dir, is_number
from kge.util.dump import add_dump_parsers, dump
from kge.util.io import get_checkpoint_file, load_checkpoint
from kge.util.package import package_model, add_package_parser
from kge.normal_cli import create_parser, process_meta_command, argparse_bool_type
from kge.distributed import WorkerProcessPool, TorchParameterServer, WorkScheduler

from torch import multiprocessing as mp
from torch import distributed as dist


def init_lapse_scheduler(servers, num_keys, master_ip, master_port, lapse_port, dist_world_size):
    # we are only initializing dist here to have the same ranks for lapse and torch
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = master_port
    print("before init mock process, world_size", dist_world_size)
    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=dist_world_size, rank=0,
    )
    print("after init mock")
    os.environ["DMLC_NUM_WORKER"] = "0"
    os.environ["DMLC_NUM_SERVER"] = str(servers)
    os.environ["DMLC_ROLE"] = "scheduler"
    os.environ["DMLC_PS_ROOT_URI"] = master_ip
    os.environ["DMLC_PS_ROOT_PORT"] = lapse_port
    num_workers_per_server = 1
    lapse.scheduler(num_keys, num_workers_per_server)


def init_torch_server(num_clients, num_keys, dim, master_ip, master_port):
    world_size = num_clients + 2
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=world_size, rank=0,
    )
    TorchParameterServer(world_size, num_keys, dim)


def main():
    # default config
    config = Config()

    # now parse the arguments
    parser = create_parser(config)
    args, unknown_args = parser.parse_known_args()

    # If there where unknown args, add them to the parser and reparse. The correctness
    # of these arguments will be checked later.
    if len(unknown_args) > 0:
        parser = create_parser(
            config, filter(lambda a: a.startswith("--"), unknown_args)
        )
        args = parser.parse_args()

    # process meta-commands
    process_meta_command(args, "create", {"command": "start", "run": False})
    process_meta_command(args, "eval", {"command": "resume", "job.type": "eval"})
    process_meta_command(
        args, "test", {"command": "resume", "job.type": "eval", "eval.split": "test"}
    )
    process_meta_command(
        args, "valid", {"command": "resume", "job.type": "eval", "eval.split": "valid"}
    )
    # dump command
    if args.command == "dump":
        dump(args)
        exit()

    # package command
    if args.command == "package":
        package_model(args)
        exit()

    # start command
    if args.command == "start":
        # use toy config file if no config given
        if args.config is None:
            args.config = kge_base_dir() + "/" + "examples/toy-complex-train.yaml"
            print(
                "WARNING: No configuration specified; using " + args.config,
                file=sys.stderr,
            )

        if args.verbose != False:
            print("Loading configuration {}...".format(args.config))
        config.load(args.config)

    # resume command
    if args.command == "resume":
        if os.path.isdir(args.config) and os.path.isfile(args.config + "/config.yaml"):
            args.config += "/config.yaml"
        if args.verbose != False:
            print("Resuming from configuration {}...".format(args.config))
        config.load(args.config)
        config.folder = os.path.dirname(args.config)
        if not config.folder:
            config.folder = "."
        if not os.path.exists(config.folder):
            raise ValueError(
                "{} is not a valid config file for resuming".format(args.config)
            )

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in [
            "command",
            "config",
            "run",
            "folder",
            "checkpoint",
            "abort_when_cache_outdated",
        ]:
            continue
        if value is not None:
            if key == "search.device_pool":
                value = "".join(value).split(",")
            try:
                if isinstance(config.get(key), bool):
                    value = argparse_bool_type(value)
            except KeyError:
                pass
            config.set(key, value)
            if key == "model":
                config._import(value)

    # initialize output folder
    if args.command == "start":
        if args.folder is None:  # means: set default
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            config.folder = os.path.join(
                kge_base_dir(),
                "local",
                "experiments",
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + config_name,
            )
        else:
            config.folder = args.folder

    # catch errors to log them
    try:
        if args.command == "start" and not config.init_folder():
            raise ValueError("output folder {} exists already".format(config.folder))
        config.log("Using folder: {}".format(config.folder))

        # determine checkpoint to resume (if any)
        if hasattr(args, "checkpoint"):
            checkpoint_file = get_checkpoint_file(config, args.checkpoint)

        # disable processing of outdated cached dataset files globally
        Dataset._abort_when_cache_outdated = args.abort_when_cache_outdated

        # log configuration
        config.log("Configuration:")
        config.log(yaml.dump(config.options), prefix="  ")
        config.log("git commit: {}".format(get_git_revision_short_hash()), prefix="  ")

        # set random seeds
        if config.get("random_seed.python") > -1:
            import random

            random.seed(config.get("random_seed.python"))
        if config.get("random_seed.torch") > -1:
            import torch

            torch.manual_seed(config.get("random_seed.torch"))
        if config.get("random_seed.numpy") > -1:
            import numpy.random

            numpy.random.seed(config.get("random_seed.numpy"))

        # let's go
        if args.command == "start" and not args.run:
            config.log("Job created successfully.")
        else:
            # load data
            dataset = Dataset.create(config)

            checkpoint = None
            # let's go
            if args.command == "resume":
                if checkpoint_file is not None:
                    checkpoint = load_checkpoint(
                        checkpoint_file, config.get("job.device")
                    )
                #     job = Job.create_from(
                #         checkpoint, new_config=config, dataset=dataset
                #     )
                # else:
                #     job = Job.create(config, dataset)
                #     job.config.log(
                #         "No checkpoint found or specified, starting from scratch..."
                #     )
            # else:
            processes = []
            num_keys = dataset.num_entities() + dataset.num_relations()
            num_meta_keys = 2
            num_workers = config.get("job.distributed.num_workers")
            master_ip = config.get("job.distributed.master_ip")
            master_port = config.get("job.distributed.master_port")
            lapse_port = config.get("job.distributed.lapse_port")
            num_partitions = config.get("job.distributed.num_partitions")
            dist_world_size = num_workers + 2
            dim = config.get("lookup_embedder.dim")
            if config.get("train.optimizer") == "dist_adagrad":
                num_keys *= 2
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
            if config.get("job.distributed.parameter_server") == "lapse":
                p = mp.Process(
                    target=init_lapse_scheduler,
                    args=(num_workers, num_keys, master_ip, master_port, lapse_port, dist_world_size),
                    daemon=True,
                )
                p.start()
                processes.append(p)
            else:
                p = mp.Process(
                    target=init_torch_server,
                    args=(num_workers, num_keys, dim, master_ip, master_port),
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
            )
            scheduler.start()
            processes.append(scheduler)
            num_workers = config.get("job.distributed.num_workers")
            worker_process_pool = WorkerProcessPool(
                num_workers, num_keys, num_meta_keys, dim, config, dataset, checkpoint
            )
            worker_process_pool.join()
            for p in processes:
                p.join()

            # job.run()
    except BaseException as e:
        tb = traceback.format_exc()
        config.log(tb, echo=False)
        raise e from None


if __name__ == "__main__":
    main()
