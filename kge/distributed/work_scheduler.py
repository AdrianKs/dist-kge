import os
import math
import datetime
import time
import concurrent.futures
import numpy as np
import pandas as pd
import numba
import torch
from collections import deque, OrderedDict
from copy import deepcopy
from kge.misc import set_seeds
from kge.distributed.two_d_block_schedule_creator import TwoDBlockScheduleCreator
from torch import multiprocessing as mp
from torch import distributed as dist
from enum import IntEnum
from typing import Optional, Dict, Tuple, List
from .misc import get_min_rank


class SCHEDULER_CMDS(IntEnum):
    GET_WORK = 0
    WORK_DONE = 1
    WORK = 2
    NO_WORK = 3
    WAIT = 4
    BARRIER = 5
    SHUTDOWN = 6
    INIT_INFO = 7
    GET_INIT_WORK = 8
    PRE_LOCALIZE_WORK = 9


class WorkScheduler(mp.get_context("spawn").Process):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset,
        dataset_folder,
        repartition_epoch=True,
    ):
        self._config_check(config)
        super(WorkScheduler, self).__init__(daemon=False, name="work-scheduler")
        self.config = config
        self.dataset = dataset
        self.rank = get_min_rank(config) - 1
        self.num_clients = num_clients
        self.world_size = world_size
        self.master_ip = master_ip
        self.master_port = master_port
        self.num_partitions = num_partitions
        self.done_workers = []
        self.asking_workers = []
        self.work_to_do = deque(list(range(num_partitions)))
        self.wait_time = 0.4
        self.repartition_epoch = repartition_epoch
        self.init_up_to_entity = -1
        if self.repartition_epoch:
            self.repartition_future = None
            self.repartition_worker_pool = None
            
    def _init_in_started_process(self):
        self.partitions = self._load_partitions(self.dataset.folder, self.num_partitions)

    def _config_check(self, config):
        if config.get("job.distributed.entity_sync_level") == "partition" and not config.get("negative_sampling.sampling_type") == "pooled":
            raise ValueError("entity sync level 'partition' only supported with 'pooled' sampling.")

    @staticmethod
    def create(
        config,
        partition_type,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset,
        dataset_folder,
        scheduling_order="random",
        repartition_epoch=True,
    ):
        if partition_type == "random_partition":
            return RandomWorkScheduler(
                config=config,
                world_size=world_size,
                master_ip=master_ip,
                master_port=master_port,
                num_partitions=num_partitions,
                num_clients=num_clients,
                dataset=dataset,
                dataset_folder=dataset_folder,
                repartition_epoch=repartition_epoch,
            )
        elif partition_type == "relation_partition":
            return RelationWorkScheduler(
                config=config,
                world_size=world_size,
                master_ip=master_ip,
                master_port=master_port,
                num_partitions=num_partitions,
                num_clients=num_clients,
                dataset_folder=dataset_folder,
                dataset=dataset,
            )
        elif partition_type == "metis_partition":
            return MetisWorkScheduler(
                config=config,
                world_size=world_size,
                master_ip=master_ip,
                master_port=master_port,
                num_partitions=num_partitions,
                num_clients=num_clients,
                dataset_folder=dataset_folder,
                dataset=dataset
            )
        elif partition_type == "2d_block_partition":
            return TwoDBlockWorkScheduler(
                config=config,
                world_size=world_size,
                master_ip=master_ip,
                master_port=master_port,
                num_partitions=num_partitions,
                num_clients=num_clients,
                dataset=dataset,
                dataset_folder=dataset_folder,
                scheduling_order=scheduling_order,
                repartition_epoch=repartition_epoch,
            )
        else:
            raise NotImplementedError()

    def run(self):
        self._init_in_started_process()
        set_seeds(config=self.config)
        os.environ["MASTER_ADDR"] = self.master_ip
        os.environ["MASTER_PORT"] = self.master_port
        # we have to have a huge timeout here, since it is only called after a complete
        #  epoch on a partition
        print("start scheduler with rank", self.rank, "world_size", self.world_size)
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=6),
        )
        barrier_count = 0
        shutdown_count = 0
        epoch_time = None
        if self.repartition_epoch:
            if self.repartition_worker_pool is None:
                self.repartition_worker_pool = concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=torch.multiprocessing.get_context("fork"),
                )
            self._repartition_in_background()

        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=torch.long)

            # refill work and distribute to all asking workers
            if len(self.done_workers) == self.num_clients:
                epoch_time += time.time()
                self.config.log(f"complete_epoch_time: {epoch_time}")
                epoch_time = None
                self._refill_work()
                for worker in self.asking_workers:
                    self._send_work(worker, cmd_buffer)
                self.done_workers = []
                self.asking_workers = []
                continue

            # fixme: this will time out if the epoch takes too long
            #  we set the timeout to 6h for now
            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            key_len = cmd_buffer[1].item()
            if cmd == SCHEDULER_CMDS.GET_WORK:
                if epoch_time is None:
                    epoch_time = -time.time()
                if rank in self.done_workers:
                    self.asking_workers.append(rank)
                    continue
                work, entities, relations, wait = self._next_work(rank)
                self._send_work(rank, cmd_buffer, work, entities, relations, wait)
            elif cmd == SCHEDULER_CMDS.WORK_DONE:
                self._handle_work_done(rank)
            elif cmd == SCHEDULER_CMDS.BARRIER:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            elif cmd == SCHEDULER_CMDS.SHUTDOWN:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down work scheduler")
                    if self.repartition_epoch:
                        if self.repartition_future is not None:
                            self.repartition_future.cancel()
                        if self.repartition_worker_pool is not None:
                            self.repartition_worker_pool.shutdown()
                    break
            elif cmd == SCHEDULER_CMDS.INIT_INFO:
                self._handle_init_info(rank)
            elif cmd == SCHEDULER_CMDS.GET_INIT_WORK:
                self._handle_get_init_work(rank=rank, embedding_layer_size=cmd_buffer[1].item())
            elif cmd == SCHEDULER_CMDS.PRE_LOCALIZE_WORK:
                work, entities, relations, wait = self._handle_pre_localize_work(rank=rank)
                self._send_work(rank, cmd_buffer, work, entities, relations, wait, pre_localize=True)
            else:
                raise ValueError(f"The work scheduler received an unknown command: {cmd}")

    def _next_work(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        raise NotImplementedError()

    def _refill_work(self):
        self.work_to_do = deque(list(range(self.num_partitions)))

    def _repartition_in_background(self):
        pass

    def _send_work(self, rank, cmd_buffer, work, entities, relations, wait, pre_localize=False):
        # work, entities, relations, wait = self._next_work(rank)
        if work is not None:
            cmd_buffer[0] = SCHEDULER_CMDS.WORK
            cmd_buffer[1] = len(work)
            dist.send(cmd_buffer, dst=rank)
            dist.send(work, dst=rank)
            if entities is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(entities)
                dist.send(cmd_buffer, dst=rank)
                dist.send(entities, dst=rank)
            if relations is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(relations)
                dist.send(cmd_buffer, dst=rank)
                dist.send(relations, dst=rank)
        elif wait:
            cmd_buffer[0] = SCHEDULER_CMDS.WAIT
            cmd_buffer[1] = self.wait_time
            dist.send(cmd_buffer, dst=rank)
        else:
            if not pre_localize:
                self.done_workers.append(rank)
            cmd_buffer[0] = SCHEDULER_CMDS.NO_WORK
            cmd_buffer[1] = 0
            dist.send(cmd_buffer, dst=rank)

    def _handle_work_done(self, rank):
        pass

    def _handle_init_info(self, rank):
        max_entities = self._get_max_entities()
        max_relations = self._get_max_relations()
        init_data = torch.LongTensor([max_entities, max_relations])
        dist.send(init_data, dst=rank)

    def _handle_get_init_work(self, rank, embedding_layer_size):
        if self.init_up_to_entity == -1:
            print("initialize parameter server")
        self.init_up_to_entity += 1
        if self.init_up_to_entity >= self.dataset.num_entities():
            return_buffer = torch.LongTensor([-1, -1])
        else:
            entity_range_end = min(self.dataset.num_entities(),
                                   self.init_up_to_entity + embedding_layer_size)
            if entity_range_end == self.dataset.num_entities():
                print("parameter server initialized")
            return_buffer = torch.LongTensor(
                [self.init_up_to_entity,
                 entity_range_end
                 ]
            )
        self.init_up_to_entity += embedding_layer_size
        dist.send(return_buffer, dst=rank)

    def _handle_pre_localize_work(self, rank):
        raise ValueError("The current partition scheme does not support pre-localizing")

    def _get_max_entities(self):
        return 0

    def _get_max_relations(self):
        return 0

    @staticmethod
    def _load_partition_file(partition_type, dataset_folder, num_partitions):
        print("loading partitions")
        # todo: we should probably build a clean variant of this in the dataset object
        if os.path.exists(
            os.path.join(
                dataset_folder,
                partition_type,
                f"num_{num_partitions}",
                "train_assign_partitions.del.npy",
            )
        ):
            partition_assignment = np.load(
                os.path.join(
                    dataset_folder,
                    partition_type,
                    f"num_{num_partitions}",
                    "train_assign_partitions.del.npy",
                )
            )
        else:
            partition_assignment = pd.read_csv(
                os.path.join(
                    dataset_folder,
                    partition_type,
                    f"num_{num_partitions}",
                    "train_assign_partitions.del"
                ),
                header=None,
                sep="\t",
                dtype=np.long,
            ).to_numpy()
            np.save(
                os.path.join(
                    dataset_folder,
                    partition_type,
                    f"num_{num_partitions}",
                    "train_assign_partitions.del.npy",
                ),
                partition_assignment,
            )
        return partition_assignment

    def _load_entities_to_partitions_file(
        self, partition_type, dataset_folder, num_partitions
    ):
        return self._load_partition_mapper_file(
            "entity_to_partitions.del", partition_type, dataset_folder, num_partitions
        )

    def _load_relations_to_partitions_file(
        self, partition_type, dataset_folder, num_partitions
    ):
        return self._load_partition_mapper_file(
            "relation_to_partitions.del", partition_type, dataset_folder, num_partitions
        )

    @staticmethod
    def _load_partition_mapper_file(
        file_name, partition_type, dataset_folder, num_partitions
    ):
        if os.path.exists(
            os.path.join(
                dataset_folder,
                partition_type,
                f"num_{num_partitions}",
                f"{file_name}.npy",
            )
        ):
            partition_assignment = np.load(
                os.path.join(
                    dataset_folder,
                    partition_type,
                    f"num_{num_partitions}",
                    f"{file_name}.npy",
                )
            )
        else:
            partition_assignment = pd.read_csv(
                os.path.join(
                    dataset_folder, partition_type, f"num_{num_partitions}", file_name,
                ),
                header=None,
                sep="\t",
                dtype=np.long,
            ).to_numpy()
            np.save(
                os.path.join(
                    dataset_folder, partition_type, f"num_{num_partitions}", file_name,
                ),
                partition_assignment,
            )
        return partition_assignment


class RandomWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset,
        dataset_folder,
        repartition_epoch,
    ):
        self.partition_type = "random_partition"
        self.dataset = dataset
        super(RandomWorkScheduler, self).__init__(
            config=config,
            world_size=world_size,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_clients,
            dataset_folder=dataset_folder,
            dataset=dataset,
            repartition_epoch=repartition_epoch,
        )

    def _next_work(
            self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """add work/partitions to the list of work to do"""
        try:
            return self.partitions[self.work_to_do.pop()], None, None, False
        except IndexError:
            return None, None, None, False

    def _load_partitions(self, dataset_folder, num_partitions):
        num_triples = len(self.dataset.split("train"))
        permuted_triple_index = torch.from_numpy(np.random.permutation(num_triples))
        partitions = list(torch.chunk(permuted_triple_index, num_partitions))
        partitions = [p.clone() for p in partitions]
        return partitions

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions = self._load_partitions(None, self.num_partitions)
        super(RandomWorkScheduler, self)._refill_work()


class RelationWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset_folder,
        dataset,
    ):
        self.partition_type = "relation_partition"
        super(RelationWorkScheduler, self).__init__(
            config=config,
            world_size=world_size,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_clients,
            dataset_folder=dataset_folder,
            dataset=dataset,
        )

    def _init_in_started_process(self):
        super(RelationWorkScheduler, self)._init_in_started_process()
        self.relations_to_partition = self._load_relations_to_partitions_file(
            self.partition_type, self.dataset.folder, self.num_partitions
        )
        self.relations_to_partition = self._get_relations_in_partition()

    def _next_work(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """add work/partitions to the list of work to do"""
        try:
            partition = self.work_to_do.pop()
            partition_data = self.partitions[partition]
            relations = self.relations_to_partition[partition]
            return partition_data, None, relations, False
        except IndexError:
            return None, None, None, False

    def _load_partitions(self, dataset_folder, num_partitions):
        partition_assignment = self._load_partition_file(
            self.partition_type, dataset_folder, num_partitions
        )
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0])
            for i in partition_indexes
        ]
        return partitions

    def _get_relations_in_partition(self):
        relations_in_partition = dict()
        for partition in range(self.num_partitions):
            relations_in_partition[partition] = torch.from_numpy(
                np.where((self.relations_to_partition == partition),)[0]
            )
        return relations_in_partition


class MetisWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset_folder,
        dataset,
    ):
        self.partition_type = "metis_partition"
        super(MetisWorkScheduler, self).__init__(
            config=config,
            world_size=world_size,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_clients,
            dataset_folder=dataset_folder,
            dataset=dataset,
        )

    def _init_in_started_process(self):
        super(MetisWorkScheduler, self)._init_in_started_process()
        self.entities_to_partition = self._load_entities_to_partitions_file(
            self.partition_type, self.dataset.folder, self.num_partitions
        )
        self.entities_to_partition = self._get_entities_in_partition()

    def _config_check(self, config):
        super(MetisWorkScheduler, self)._config_check(config)
        if config.get("job.distributed.entity_sync_level") == "partition":
            raise ValueError("Metis partitioning does not support entity sync level 'parititon'. "
                             "Triples still have outside partition accesses.")

    def _next_work(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """add work/partitions to the list of work to do"""
        try:
            partition = self.work_to_do.pop()
            partition_data = self.partitions[partition]
            entities = self.entities_to_partition[partition]
            return partition_data, entities, None, False
        except IndexError:
            return None, None, None, False

    def _load_partitions(self, dataset_folder, num_partitions):
        partition_assignment = self._load_partition_file(
            self.partition_type, dataset_folder, num_partitions
        )
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0])
            for i in partition_indexes
        ]
        return partitions

    def _get_entities_in_partition(self):
        entities_in_partition = dict()
        for partition in range(self.num_partitions):
            entities_in_partition[partition] = torch.from_numpy(
                np.where((self.entities_to_partition == partition),)[0]
            )
        return entities_in_partition

    def _get_max_entities(self):
        return max([len(i) for i in self.entities_to_partition.values()])


class TwoDBlockWorkScheduler(WorkScheduler):
    """
    Lets look at the PBG scheduling here to make it correct
    """

    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset,
        dataset_folder,
        scheduling_order="random",
        repartition_epoch=True,
    ):
        self.partition_type = "2d_block_partition"
        self.combine_mirror_blocks = config.get("job.distributed.combine_mirror_blocks")
        self.schedule_creator = TwoDBlockScheduleCreator(
            num_partitions=num_partitions,
            num_workers=num_clients,
            randomize_iterations=True,
            combine_mirror_blocks=self.combine_mirror_blocks
        )
        #self.fixed_schedule = [item for sublist in self.schedule_creator.create_schedule() for item in sublist]
        self.fixed_schedule = self.schedule_creator.create_schedule()
        super(TwoDBlockWorkScheduler, self).__init__(
            config=config,
            world_size=world_size,
            master_ip=master_ip,
            master_port=master_port,
            num_partitions=num_partitions,
            num_clients=num_clients,
            dataset_folder=dataset_folder,
            dataset=dataset,
            repartition_epoch=repartition_epoch,
        )
        self.entities_needed_only = self.config.get(
            "job.distributed.stratification.entities_needed_only"
        )
        self.scheduling_order = scheduling_order
        self.num_max_entities = 0
        
    def _init_in_started_process(self):
        super(TwoDBlockWorkScheduler, self)._init_in_started_process()
        # dictionary: key=worker_rank, value=block
        self.running_blocks: Dict[int, Tuple[int, int]] = {}
        # self.work_to_do = deepcopy(self.partitions)
        self._initialized_entity_blocks = set()
        entities_to_partition = self._load_entities_to_partitions_file(
            self.partition_type, self.dataset.folder, self.num_partitions
        )
        self._entities_in_bucket = self._get_entities_in_bucket(
            entities_to_partition,
            self.partitions,
            self.dataset.split("train"),
            self.entities_needed_only
        )
        self.work_to_do: Dict[Tuple[int, int], torch.Tensor] = self._order_by_schedule(
            deepcopy(self.partitions)
        )
        self.current_iteration = set()
        self._pre_localized_strata: Dict[int, Tuple[int, int]] = {}

    def _order_by_schedule(
        self, partitions: Dict[Tuple[int, int], torch.Tensor]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        sorted_partitions_keys = [0] * (len(partitions.keys()))
        num_entity_blocks = int(math.sqrt(len(partitions)))
        if self.scheduling_order == "sequential":
            # 00 11 22 01 12 20 02 10 21
            sorted_partitions_keys = []
            for i in range(num_entity_blocks):
                sorted_partitions_keys.append((i, i))
            for i in range(num_entity_blocks):
                for j in range(num_entity_blocks):
                    object_id = (j + i + 1) % num_entity_blocks
                    if object_id == j:
                        continue
                    sorted_partitions_keys.append((j, object_id))
        elif self.scheduling_order == "sequential_old":
            for bucket in partitions.keys():
                if bucket[0] == bucket[1]:
                    sorted_partitions_keys[bucket[0].item()] = bucket
                else:
                    position = (num_entity_blocks * (bucket[0] + 1)) + bucket[1]
                    position -= bucket[0]
                    if bucket[1] > bucket[0]:
                        position -= 1
                    sorted_partitions_keys[position.item()] = bucket
        elif self.scheduling_order == "random":
            positions = np.random.permutation(np.arange(len(partitions.keys())))
            for i, bucket in zip(positions, partitions.keys()):
                sorted_partitions_keys[i] = bucket
        elif self.scheduling_order == "inside_out":
            # 00 01 10 11 02 20 12 21 22
            sorted_partitions_keys = []
            for i in range(num_entity_blocks):
                sorted_partitions_keys.append((i, i))
                if i == num_entity_blocks - 1:
                    break
                for j in range(i):
                    sorted_partitions_keys.append((j, i + 1))
                    sorted_partitions_keys.append((i + 1, j))
                sorted_partitions_keys.append((i, i + 1))
                sorted_partitions_keys.append((i + 1, i))
        else:
            raise NotImplementedError()
        sorted_partitions = OrderedDict()
        for key in sorted_partitions_keys:
            sorted_partitions[key] = partitions[key]
        return sorted_partitions

    @staticmethod
    @numba.guvectorize([(numba.int64[:], numba.int64, numba.int64, numba.int64[:])], '(n),(),()->(n)')
    def _get_partition(entity_ids, num_entities, num_partitions, res):
        """
        This method maps a (already mapped) entity id to it's entity_partition.
        NOTE: you can not provide named parameters (kwargs) to this function
        Args:
            entity_ids: (mapped) entity ids np.array()
            num_entities: dataset.num_entities()
            num_partitions: int
            res: DON'T PROVIDE THIS. This is the resulting np.array of this vectorized
                function.

        Returns: np.array of entity ids mapped to partition

        """
        for i in range(len(entity_ids)):
            res[i] = math.floor(
                entity_ids[i] * 1.0 / num_entities * 1.0 * num_partitions
            )

    @staticmethod
    def _repartition(data, num_entities, num_partitions, entities_needed_only=True):
        """
        This needs to be a static method so that we can pickle and run in background
        Args:
            data: data to repartition (train-set)
            num_entities: dataset.num_entities()
            num_partitions: self.num_partitions

        Returns:
            partitions: dict of structure {(block_id 1, block_id 2): [triple ids]}
            entities_in_bucket:
                dict of structure {(block_id 1, block_id 2): list of entity ids}
        """
        print("repartitioning data")
        start = -time.time()

        def random_map_entities():
            mapper = np.random.permutation(num_entities)
            mapped_data = deepcopy(
                data
            )  # drop reference to dataset
            mapped_data = mapped_data.numpy()
            mapped_data[:, 0] = mapper[mapped_data[:, 0]]
            mapped_data[:, 2] = mapper[mapped_data[:, 2]]
            return mapped_data, mapper

        mapped_data, mapped_entities = random_map_entities()
        print("repartition s")
        s_block = TwoDBlockWorkScheduler._get_partition(
            mapped_data[:, 0],
            num_entities,
            num_partitions,
        )
        print("repartition o")
        o_block = TwoDBlockWorkScheduler._get_partition(
            mapped_data[:, 2],
            num_entities,
            num_partitions,
        )
        print("map entity ids to partition")
        entity_to_partition = TwoDBlockWorkScheduler._get_partition(
            mapped_entities,
            num_entities,
            num_partitions,
        )
        triple_partition_assignment = np.stack([s_block, o_block], axis=1)
        partitions = TwoDBlockWorkScheduler._construct_partitions(
            triple_partition_assignment,
            num_partitions
        )
        entities_in_bucket = TwoDBlockWorkScheduler._get_entities_in_bucket(
            entity_to_partition,
            partitions,
            data.numpy(),
            entities_needed_only
        )
        print("repartitioning done")
        print("repartition_time", start+time.time())
        return partitions, entities_in_bucket

    @staticmethod
    def _get_entities_in_bucket(entities_to_partition, partitions, data, entities_needed_only):
        entities_in_bucket = dict()
        if entities_needed_only:
            for strata, strata_data in partitions.items():
                # np.unique is slightly faster than torch.unique
                entities_in_bucket[strata] = torch.from_numpy(
                    np.unique(data[strata_data][:, [0, 2]]).astype(np.long)
                ).contiguous()
        else:
            for partition in partitions:
                entities_in_bucket[partition] = torch.from_numpy(
                    np.where(
                        np.ma.mask_or(
                            (entities_to_partition == partition[0]),
                            (entities_to_partition == partition[1]),
                        )
                    )[0]
                ).contiguous()
        return entities_in_bucket

    def _get_max_entities(self):
        if self.num_max_entities > 0:
            # store the result so that we don't have to recompute for every trainer
            return self.num_max_entities
        if self.entities_needed_only:
            num_entities_in_strata = [len(i) for i in self._entities_in_bucket.values()]
            len_std = np.std(num_entities_in_strata).item()
            if self.combine_mirror_blocks:
                max_num_entities, std_num_entities = self._get_mirrored_max_entities(
                    self.num_partitions,
                    list(self._entities_in_bucket.values()),
                    return_std=True
                )
                self.num_max_entities = max_num_entities + 2*(round(std_num_entities))
            else:
                self.num_max_entities = max(num_entities_in_strata) + 5*round(len_std)
        else:
            self.num_max_entities = max(
                [len(i) for i in self._entities_in_bucket.values()]
            )
        return self.num_max_entities

    @staticmethod
    def _get_mirrored_max_entities(num_partitions, strata_entities, return_std=False):
        """
        Calculate how many entities occur at most if we combine mirrored blocks
        Combining blocks (0,1) and (1,0)
        For diagonals combine (0,0),(1,1), then (2,2),(3,3)...
        Count unique entities per combined block and return max
        Args:
            num_partitions: number of partitions
            strata_entities: list of unique entities occurring per strata
                assumes list is ordered

        Returns: max number of entities occurring in a combined mirror block

        """
        max_value = 0
        all_num_entities = []
        for i in range(num_partitions):
            for j in range(i, num_partitions):
                num_entities = 0
                # combine mirrored blocks
                if i % 2 == 0 and i == j:
                    # diagonal blocks: combine with following diagonal
                    concat_entities = np.concatenate(
                        (strata_entities[i], strata_entities[i+num_partitions])
                    )
                    num_entities = len(np.unique(concat_entities))
                elif i != j:
                    # combine (0,1) with (1,0) and so on
                    num_entities = len(np.unique(np.concatenate(
                        (strata_entities[i*num_partitions+j],
                         strata_entities[j*num_partitions+i]))
                    ))
                all_num_entities.append(num_entities)
                if num_entities > max_value:
                    # this will lead to a race condition if we do this in parallel
                    max_value = num_entities
        all_num_entities = np.array(all_num_entities)
        max_value = all_num_entities.max()
        if return_std:
            std = all_num_entities.std()
            return max_value, std
        print("max entities", max_value)
        return max_value

    def _next_work(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        if self.fixed_schedule is not None:
            return self._acquire_bucket_by_fixed_schedule(rank)
        return self._acquire_bucket(rank)

    def _handle_pre_localize_work(self, rank):
        return self._acquire_bucket_by_fixed_schedule(rank, pre_localize=True)

    def _acquire_bucket_by_fixed_schedule(self, rank, pre_localize=False):
        try:
            locked_entity_strata = set()
            for locked_dict in [self.running_blocks, self._pre_localized_strata]:
                for running_rank, strata in locked_dict.items():
                    if rank == running_rank:
                        continue
                    locked_entity_strata.add(strata[0])
                    locked_entity_strata.add(strata[1])

            def _strata_locked(strata):
                return strata[0] in locked_entity_strata or strata[1] in locked_entity_strata

            def _acquire(strata, acquire_pre_localized=False):
                if acquire_pre_localized:
                    del self._pre_localized_strata[rank]
                else:
                    self.current_iteration.remove(strata)
                strata_data = self.partitions[strata]
                entities_in_strata = self._entities_in_bucket.get(strata)
                if self.combine_mirror_blocks:
                    if strata[0] == strata[1]:
                        mirror_strata = (strata[0]-1, strata[1]-1)
                        entities_in_strata = torch.cat(
                            (entities_in_strata,
                             self._entities_in_bucket.get(mirror_strata))
                        )
                    else:
                        mirror_strata = (strata[1], strata[0])
                        if self.entities_needed_only:
                            entities_in_strata = torch.unique(torch.cat(
                                (entities_in_strata,
                                 self._entities_in_bucket.get(mirror_strata))
                            ))
                    strata_data = torch.cat(
                        (strata_data, self.partitions[mirror_strata])
                    )
                if not pre_localize:
                    self.running_blocks[rank] = strata
                else:
                    self._pre_localized_strata[rank] = strata
                return strata_data, entities_in_strata, None, False

            # only use pre localized strata, if we are not about to pre-localize a new
            # one --> not pre_localize
            if not pre_localize and self._pre_localized_strata.get(rank, None) is not None:
                strata = self._pre_localized_strata[rank]
                if _strata_locked(strata):
                    # we are waiting until the localized strata is free
                    return None, None, None, True
                return _acquire(strata, acquire_pre_localized=True)

            if len(self.current_iteration) == 0:
                self.current_iteration = set(self.fixed_schedule.pop())

            for strata in self.current_iteration:
                if _strata_locked(strata):
                    continue
                return _acquire(strata)

            # return wait here
            return None, None, None, True
        except IndexError:
            return None, None, None, False

    def _acquire_bucket(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """
        Finds a (lhs, rhs) partition pair that has not already been acquired
        this epoch, and where neither the lhs nor rhs partitions are currently
        locked. Locks this lhs and rhs until `release_pair` is called. Will try
        to find a pair that has the same lhs (if not, rhs) as old_bucket.

        If no pair is available, returns None.

        Returns:
            pair: a (lhs, rhs) partition pair. lhs and rhs are locked until
                  `release_pair` is called.
                  If no pair is available, None is returned.
            remaining: The number of pairs remaining. When this is 0 then the
                       epoch is done.
        """
        wait = False
        locked_entity_blocks = {}
        for worker_rank, bucket in self.running_blocks.items():
            locked_entity_blocks[bucket[0]] = (bucket, worker_rank)
            locked_entity_blocks[bucket[1]] = (bucket, worker_rank)

        acquirable_entity_blocks = [
            i
            for i in range(self.num_partitions)
            if i not in locked_entity_blocks.keys()
        ]
        acquirable_entity_blocks = set(acquirable_entity_blocks)

        def _is_acquirable(bucket: Tuple[int, int]):
            if not self._is_initialized(bucket):
                return False
            if not bucket[0] in acquirable_entity_blocks:
                return False
            if not bucket[1] in acquirable_entity_blocks:
                return False
            return True

        for block, block_data in self.work_to_do.items():
            if _is_acquirable(block):
                self.running_blocks[rank] = block
                self._initialized_entity_blocks.add(block[0])
                self._initialized_entity_blocks.add(block[1])
                del self.work_to_do[block]
                return block_data, self._entities_in_bucket.get(block), None, False
        if len(self.work_to_do) > 0:
            wait = True
        return None, None, None, wait

    def _is_initialized(self, bucket: Tuple[int, int]):
        # at least one side of the partition block needs to be initialized to ensure
        #  all embeddings are in the same embedding space
        #  if nothing is initialized start with anything
        if len(self._initialized_entity_blocks) == 0:
            return True
        return (
            bucket[0] in self._initialized_entity_blocks
            or bucket[1] in self._initialized_entity_blocks
        )

    def _handle_work_done(self, rank):
        del self.running_blocks[rank]

    def _repartition_in_background(self):
        self.repartition_future = self.repartition_worker_pool.submit(
            self._repartition,
            self.dataset.split("train"),
            self.dataset.num_entities(),
            self.num_partitions,
            self.entities_needed_only
        )

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions, self._entities_in_bucket = self.repartition_future.result()
            self._repartition_in_background()
        #self.fixed_schedule = [item for sublist in self.schedule_creator.create_schedule() for item in sublist]
        self.fixed_schedule = self.schedule_creator.create_schedule()
        self.work_to_do = self._order_by_schedule(deepcopy(self.partitions))

    def _load_partitions(self, dataset_folder, num_partitions):
        start = time.time()
        partition_assignment = self._load_partition_file(
            self.partition_type, dataset_folder, num_partitions
        )
        partitions = self._construct_partitions(partition_assignment, num_partitions)
        print("partition load time", time.time()-start)
        return partitions

    @staticmethod
    def _construct_partitions(partition_assignment, num_partitions):
        partition_indexes, partition_data = TwoDBlockWorkScheduler._numba_construct_partitions(np.ascontiguousarray(partition_assignment), num_partitions)
        partition_indexes = [(i, j) for i in range(num_partitions) for j in range(num_partitions)]
        partition_data = [torch.from_numpy(data).long().contiguous() for data in partition_data]
        partitions = dict(zip(partition_indexes, partition_data))
        return partitions

    @staticmethod
    @numba.njit
    def _numba_construct_partitions(partition_assignment, num_partitions):
        partition_indexes = [
            (i, j) for i in range(num_partitions) for j in range(num_partitions)
        ]
        partition_id_lookup: Dict[Tuple[int, int], int] = dict()
        partition_lengths: Dict[int, int] = dict()
        partition_data = []
        for i in range(len(partition_indexes)):
            partition = partition_indexes[i]
            partition_id_lookup[partition] = i
            partition_lengths[i] = 0
            partition_data.append(
                np.empty(
                    int(len(partition_assignment)/((num_partitions*num_partitions)/2)),
                    dtype=np.int64
                )
            )

        # iterate over the partition assignments and assign each triple-id to its
        #  corresponding partition
        for i in range(len(partition_assignment)):
            pa = partition_assignment[i]
            pa_tuple = (pa[0], pa[1])
            partition_id = partition_id_lookup[pa_tuple]
            current_partition_size = partition_lengths[partition_id]
            partition_data[partition_id][current_partition_size] = i
            partition_lengths[partition_id] += 1

        # now get correct sizes of partitions
        for i in range(len(partition_data)):
            partition_data[i] = partition_data[i][:partition_lengths[i]]
        return partition_indexes, partition_data


class SchedulerClient:
    def __init__(self, config):
        self.scheduler_rank = get_min_rank(config) - 1

    def get_init_info(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.INIT_INFO, 0])
        dist.send(cmd, dst=self.scheduler_rank)
        info_buffer = torch.zeros((2,), dtype=torch.long)
        dist.recv(info_buffer, src=self.scheduler_rank)
        max_entities = info_buffer[0]
        max_relations = info_buffer[1]
        return max_entities, max_relations

    def _receive_work(self, cmd):
        work_buffer = torch.empty((cmd[1].item(),), dtype=torch.long)
        dist.recv(work_buffer, src=self.scheduler_rank)
        # get partition entities
        dist.recv(cmd, src=self.scheduler_rank)
        num_entities = cmd[1].item()
        entity_buffer = None
        if num_entities != 0:
            entity_buffer = torch.empty((num_entities,), dtype=torch.long)
            dist.recv(entity_buffer, src=self.scheduler_rank)
        # get partition relations
        dist.recv(cmd, src=self.scheduler_rank)
        num_relations = cmd[1].item()
        relation_buffer = None
        if num_relations != 0:
            relation_buffer = torch.empty((num_relations,), dtype=torch.long)
            dist.recv(relation_buffer, src=self.scheduler_rank)
        return work_buffer, entity_buffer, relation_buffer

    def get_work(
            self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        while True:
            cmd = torch.LongTensor([SCHEDULER_CMDS.GET_WORK, 0])
            dist.send(cmd, dst=self.scheduler_rank)
            dist.recv(cmd, src=self.scheduler_rank)
            if cmd[0] == SCHEDULER_CMDS.WORK:
                return self._receive_work(cmd)
            elif cmd[0] == SCHEDULER_CMDS.WAIT:
                # print("waiting for a block")
                time.sleep(cmd[1].item())
            else:
                return None, None, None

    def get_pre_localize_work(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.PRE_LOCALIZE_WORK, 0])
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] == SCHEDULER_CMDS.WORK:
            work, entities, relations = self._receive_work(cmd)
            return work, entities, relations, False
        elif cmd[0] == SCHEDULER_CMDS.WAIT:
            return None, None, None, True
        else:
            return None, None, None, False

    def get_init_work(self, entity_embedder_size):
        """
        Get the entity ids that should be initialized by the worker.
        Receives start and end id from the scheduler
        Args:
            entity_embedder_size: size of the local entity embedding layer

        Returns:
            tensor containing range from start and end entity id

        """
        cmd = torch.LongTensor([SCHEDULER_CMDS.GET_INIT_WORK, entity_embedder_size])
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] > -1:
            return torch.arange(cmd[0], cmd[1], dtype=torch.long)
        return None

    def work_done(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.WORK_DONE, 0])
        dist.send(cmd, dst=self.scheduler_rank)

    def shutdown(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.SHUTDOWN, 0])
        dist.send(cmd, dst=self.scheduler_rank)
