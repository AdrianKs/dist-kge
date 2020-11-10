import os
import math
import datetime
import time
import concurrent.futures
import numpy as np
import torch
from collections import deque, OrderedDict
from copy import deepcopy
from kge.misc import set_seeds
from kge.distributed.two_d_block_schedule_creator import TwoDBlockScheduleCreator
from torch import multiprocessing as mp
from torch import distributed as dist
from enum import IntEnum
from typing import Optional, Dict, Tuple, List


class SCHEDULER_CMDS(IntEnum):
    GET_WORK = 0
    WORK_DONE = 1
    WORK = 2
    NO_WORK = 3
    WAIT = 4
    BARRIER = 5
    SHUTDOWN = 6
    INIT_INFO = 7


class WorkScheduler(mp.get_context("spawn").Process):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset_folder,
        rank=1,
        repartition_epoch=True,
    ):
        self._config_check(config)
        super(WorkScheduler, self).__init__(daemon=False, name="work-scheduler")
        self.config = config
        self.rank = rank
        self.num_clients = num_clients
        self.world_size = world_size
        self.master_ip = master_ip
        self.master_port = master_port
        self.num_partitions = num_partitions
        self.partitions = self._load_partitions(dataset_folder, num_partitions)
        self.done_workers = []
        self.asking_workers = []
        self.work_to_do = deque(list(range(num_partitions)))
        self.wait_time = 0.4
        self.repartition_epoch = repartition_epoch
        if self.repartition_epoch:
            self.repartition_future = None
            self.repartition_worker_pool = None

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
        if partition_type == "block_partition":
            return BlockWorkScheduler(
                config=config,
                world_size=world_size,
                master_ip=master_ip,
                master_port=master_port,
                num_partitions=num_partitions,
                num_clients=num_clients,
                dataset_folder=dataset_folder,
            )
        elif partition_type == "random_partition":
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
        set_seeds(config=self.config)
        os.environ["MASTER_ADDR"] = self.master_ip
        os.environ["MASTER_PORT"] = self.master_port
        # we have to have a huge timeout here, since it is only called after a complete
        #  epoch on a partition
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=2),
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
            #  we set the timeout to 2h for now
            rank = dist.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()
            key_len = cmd_buffer[1].item()
            if cmd == SCHEDULER_CMDS.GET_WORK:
                if epoch_time is None:
                    epoch_time = -time.time()
                if rank in self.done_workers:
                    self.asking_workers.append(rank)
                    continue
                self._send_work(rank, cmd_buffer)
            if cmd == SCHEDULER_CMDS.WORK_DONE:
                self._handle_work_done(rank)
            if cmd == SCHEDULER_CMDS.BARRIER:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            if cmd == SCHEDULER_CMDS.SHUTDOWN:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down work scheduler")
                    if self.repartition_epoch:
                        if self.repartition_future is not None:
                            self.repartition_future.cancel()
                    if self.repartition_worker_pool is not None:
                        self.repartition_worker_pool.shutdown()
                    break
            if cmd == SCHEDULER_CMDS.INIT_INFO:
                self._handle_init_info(rank)

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

    def _send_work(self, rank, cmd_buffer):
        work, entities, relations, wait = self._next_work(rank)
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
            partition_assignment = np.loadtxt(
                os.path.join(
                    dataset_folder,
                    partition_type,
                    f"num_{num_partitions}",
                    "train_assign_partitions.del",
                ),
                dtype=np.long,
            )
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
            partition_assignment = np.loadtxt(
                os.path.join(
                    dataset_folder, partition_type, f"num_{num_partitions}", file_name,
                ),
                dtype=np.long,
            )
            np.save(
                os.path.join(
                    dataset_folder, partition_type, f"num_{num_partitions}", file_name,
                ),
                partition_assignment,
            )
        return partition_assignment


class BlockWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        world_size,
        master_ip,
        master_port,
        num_partitions,
        num_clients,
        dataset_folder,
    ):
        self.partition_type = "block_partition"
        super(BlockWorkScheduler, self).__init__(
            config,
            world_size,
            master_ip,
            master_port,
            num_partitions,
            num_clients,
            dataset_folder,
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
            config,
            world_size,
            master_ip,
            master_port,
            num_partitions,
            num_clients,
            dataset_folder,
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
        permuted_triple_index = torch.randperm(num_triples)
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
    ):
        self.partition_type = "relation_partition"
        super(RelationWorkScheduler, self).__init__(
            config,
            world_size,
            master_ip,
            master_port,
            num_partitions,
            num_clients,
            dataset_folder,
        )
        self.relations_to_partition = self._load_relations_to_partitions_file(
            self.partition_type, dataset_folder, num_partitions
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
    ):
        self.partition_type = "metis_partition"
        super(MetisWorkScheduler, self).__init__(
            config,
            world_size,
            master_ip,
            master_port,
            num_partitions,
            num_clients,
            dataset_folder,
        )
        self.entities_to_partition = self._load_entities_to_partitions_file(
            self.partition_type, dataset_folder, num_partitions
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
        self.schedule_creator = TwoDBlockScheduleCreator(num_partitions=num_partitions, num_workers=num_clients, randomize_iterations=True)
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
            repartition_epoch=repartition_epoch,
        )
        # dictionary: key=worker_rank, value=block
        self.running_blocks: Dict[int, Tuple[int, int]] = {}
        # self.work_to_do = deepcopy(self.partitions)
        self._initialized_entity_blocks = set()
        entities_to_partition = self._load_entities_to_partitions_file(
            self.partition_type, dataset_folder, num_partitions
        )
        self._entities_in_bucket = self._get_entities_in_bucket(entities_to_partition, self.partitions)
        self.dataset = dataset
        self.scheduling_order = scheduling_order
        self.work_to_do: Dict[Tuple[int, int], torch.Tensor] = self._order_by_schedule(
            deepcopy(self.partitions)
        )
        self.current_iteration = set()

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
    def _repartition(data, num_entities, num_partitions):
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
            mapper = torch.randperm(num_entities).type(torch.int32)
            mapped_data = deepcopy(
                data
            )  # drop reference to dataset
            mapped_data[:, 0] = mapper[mapped_data[:, 0].long()]
            mapped_data[:, 2] = mapper[mapped_data[:, 2].long()]
            return mapped_data, mapper

        def get_partition(entity_id, num_entities, num_partitions):
            partition = math.floor(
                entity_id * 1.0 / num_entities * 1.0 * num_partitions
            )
            return partition

        v_get_partition = np.vectorize(
            get_partition, excluded=["num_entities", "num_partitions"]
        )
        mapped_data, mapped_entities = random_map_entities()
        print("repartition s")
        s_block = v_get_partition(
            mapped_data[:, 0],
            num_entities=num_entities,
            num_partitions=num_partitions,
        )
        print("repartition o")
        o_block = v_get_partition(
            mapped_data[:, 2],
            num_entities=num_entities,
            num_partitions=num_partitions,
        )
        print("map entity ids to partition")
        entity_to_partition = v_get_partition(
            mapped_entities,
            num_entities=num_entities,
            num_partitions=num_partitions,
        )
        triple_partition_assignment = np.stack([s_block, o_block], axis=1)
        partitions = TwoDBlockWorkScheduler._construct_partitions(triple_partition_assignment)
        entities_in_bucket = TwoDBlockWorkScheduler._get_entities_in_bucket(entity_to_partition, partitions)
        print("repartitioning done")
        print("repartition_time", start+time.time())
        return partitions, entities_in_bucket

    @staticmethod
    def _get_entities_in_bucket(entities_to_partition, partitions):
        entities_in_bucket = dict()
        for partition in partitions:
            entities_in_bucket[partition] = torch.from_numpy(
                np.where(
                    np.ma.mask_or(
                        (entities_to_partition == partition[0]),
                        (entities_to_partition == partition[1]),
                    )
                )[0]
            )
        return entities_in_bucket

    def _get_max_entities(self):
        return max([len(i) for i in self._entities_in_bucket.values()])

    def _next_work(
        self, rank
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool
    ]:
        if self.fixed_schedule is not None:
            return self._acquire_bucket_by_fixed_schedule(rank)
        return self._acquire_bucket(rank)

    def _acquire_bucket_by_fixed_schedule(self, rank):
        try:
            if len(self.current_iteration) == 0:
                self.current_iteration = set(self.fixed_schedule.pop())
            locked_entity_strata = set()
            for strata in self.running_blocks.values():
                locked_entity_strata.add(strata[0])
                locked_entity_strata.add(strata[1])
            for strata in self.current_iteration:
                if strata[0] in locked_entity_strata:
                    continue
                if strata[1] in locked_entity_strata:
                    continue
                self.current_iteration.remove(strata)
                strata_data = self.partitions[strata]
                entities_in_strata = self._entities_in_bucket.get(strata)
                self.running_blocks[rank] = strata
                return strata_data, entities_in_strata, None, False
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
            self.num_partitions
        )

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions, self._entities_in_bucket = self.repartition_future.result()
            self._repartition_in_background()
        #self.fixed_schedule = [item for sublist in self.schedule_creator.create_schedule() for item in sublist]
        self.fixed_schedule = self.schedule_creator.create_schedule()
        self.work_to_do = self._order_by_schedule(deepcopy(self.partitions))

    def _load_partitions(self, dataset_folder, num_partitions):
        partition_assignment = self._load_partition_file(
            self.partition_type, dataset_folder, num_partitions
        )
        return self._construct_partitions(partition_assignment)

    @staticmethod
    def _construct_partitions(partition_assignment):
        partition_indexes = np.unique(partition_assignment, axis=0)
        partitions_data = [
            torch.from_numpy(
                np.where(np.all(partition_assignment == i, axis=1))[0]
            ).contiguous()
            for i in partition_indexes
        ]
        partition_indexes = [(i[0], i[1]) for i in partition_indexes]
        partitions = dict(zip(partition_indexes, partitions_data))
        return partitions


class SchedulerClient:
    def __init__(self, scheduler_rank=1):
        self.scheduler_rank = scheduler_rank

    def get_init_info(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.INIT_INFO, 0])
        dist.send(cmd, dst=self.scheduler_rank)
        info_buffer = torch.zeros((2,), dtype=torch.long)
        dist.recv(info_buffer, src=self.scheduler_rank)
        max_entities = info_buffer[0]
        max_relations = info_buffer[1]
        return max_entities, max_relations

    def get_work(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        while True:
            cmd = torch.LongTensor([SCHEDULER_CMDS.GET_WORK, 0])
            dist.send(cmd, dst=self.scheduler_rank)
            dist.recv(cmd, src=self.scheduler_rank)
            if cmd[0] == SCHEDULER_CMDS.WORK:
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
            elif cmd[0] == SCHEDULER_CMDS.WAIT:
                # print("waiting for a block")
                time.sleep(cmd[1].item())
            else:
                return None, None, None

    def work_done(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.WORK_DONE, 0])
        dist.send(cmd, dst=self.scheduler_rank)

    def shutdown(self):
        cmd = torch.LongTensor([SCHEDULER_CMDS.SHUTDOWN, 0])
        dist.send(cmd, dst=self.scheduler_rank)
