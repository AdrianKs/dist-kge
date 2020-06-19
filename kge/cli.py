#!/usr/bin/env python
import datetime
import os
import sys
import traceback
import yaml

import lapse

from kge import Dataset
from kge import Config
from kge.job import Job
from kge.misc import get_git_revision_short_hash, kge_base_dir, is_number
from kge.util.dump import add_dump_parsers, dump
from kge.util.io import get_checkpoint_file, load_checkpoint
from kge.util.package import package_model, add_package_parser
from kge.normal_cli import create_parser, process_meta_command, argparse_bool_type

from copy import deepcopy
from torch import multiprocessing as mp
import time

servers = 4
num_workers_per_server = 1
localip = "127.0.0.1"
port = "9091"
mp.set_start_method("spawn", force=True)


class LapseWorker(lapse.Worker):
    def __init__(self, customer_id: int, worker_id: int, lapse_server: lapse.Server):
        super(LapseWorker, self).__init__(customer_id, worker_id, lapse_server)
        self.worker_id = worker_id


def init_scheduler(servers, num_keys):
    os.environ["DMLC_NUM_WORKER"] = "0"
    os.environ["DMLC_NUM_SERVER"] = str(servers)
    os.environ["DMLC_ROLE"] = "scheduler"
    os.environ["DMLC_PS_ROOT_URI"] = localip
    os.environ["DMLC_PS_ROOT_PORT"] = port
    lapse.scheduler(num_keys, num_workers_per_server)


def init_server(rank, servers, num_keys, embedding_dim, config, dataset):
    os.environ["DMLC_NUM_WORKER"] = "0"
    os.environ["DMLC_NUM_SERVER"] = str(servers)
    os.environ["DMLC_ROLE"] = "server"
    os.environ["DMLC_PS_ROOT_URI"] = localip
    os.environ["DMLC_PS_ROOT_PORT"] = port

    lapse.setup(num_keys, num_workers_per_server)
    s = lapse.Server(num_keys, embedding_dim)
    configs = {}
    datasets = {}
    processes = []
    start_time = time.time()
    device_pool: list = config.get("job.device_pool")
    if len(device_pool) == 0:
        device_pool.append(config.get("job.device"))
    for w in range(num_workers_per_server):
        print(num_workers_per_server)
        worker_id = rank * num_workers_per_server + w
        configs[w] = deepcopy(config)
        configs[w].set("job.device", device_pool[w % len(device_pool)])
        configs[w].folder = os.path.join(config.folder, f"worker-{w}")
        configs[w].init_folder()
        datasets[w] = deepcopy(dataset)
        datasets[w] = Dataset.create(
            configs[w], folder=os.path.join(dataset.folder, f"partition_{worker_id}")
        )
        # datasets[w] = Dataset.create(configs[w], dataset.folder)
        # kv = lapse.Worker(0, worker_id + 1, s)
        kv = LapseWorker(0, worker_id + 1, s)
        job = Job.create(configs[w], datasets[w], lapse_worker=kv)
        job.run()
        # p = threading.Thread(target=job.run)
        # p.start()
        # processes.append(p)
    for p in processes:
        p.join()
    end_time = time.time()
    print(end_time - start_time)

    # shutdown server
    s.shutdown()


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

            # let's go
            if args.command == "resume":
                if checkpoint_file is not None:
                    checkpoint = load_checkpoint(
                        checkpoint_file, config.get("job.device")
                    )
                    job = Job.create_from(
                        checkpoint, new_config=config, dataset=dataset
                    )
                else:
                    job = Job.create(config, dataset)
                    job.config.log(
                        "No checkpoint found or specified, starting from scratch..."
                    )
            else:
                configs = {}
                processes = []
                num_keys = dataset.num_entities() + dataset.num_relations()
                if config.get("train.optimizer") == "dist_adagrad":
                    num_keys *= 2
                p = mp.Process(target=init_scheduler, args=(servers, num_keys))
                p.start()
                processes.append(p)
                for rank in range(servers):
                    configs[rank] = deepcopy(config)
                    configs[rank].set(config.get("model") + ".create_complete", False)
                    configs[rank].folder = os.path.join(config.folder, f"server-{rank}")
                    configs[rank].init_folder()
                    print("before init server")
                    p = mp.Process(
                        target=init_server,
                        args=(
                            rank,
                            servers,
                            num_keys,
                            config.get("lookup_embedder.dim"),
                            configs[rank],
                            dataset,
                        ),
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            # job.run()
    except BaseException as e:
        tb = traceback.format_exc()
        config.log(tb, echo=False)
        raise e from None


if __name__ == "__main__":
    main()
