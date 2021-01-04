import time
import torch
import torch.utils.data
import numpy as np
import math
import gc
import os

from collections import deque

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
from kge.model import KgeModel
from kge.util import KgeOptimizer
from kge.util.metric import Metric
from kge.distributed.work_scheduler import SchedulerClient
from kge.distributed.misc import get_min_rank

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, dataset):
        self.samples = list(range(num_samples))
        self.dataset = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx
        # return self.dataset[self.samples[idx], :].long()

    def set_samples(self, samples):
        self.samples = samples


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, triples, batch_size, shuffle=True):
        self.triples = triples
        self.samples = None
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        if self.samples is None:
            return 0
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, idx):
        """Gets a complete batch based on an idx"""
        return self.samples[
            idx
            * self.batch_size : min((idx + 1) * (self.batch_size), len(self.samples))
        ].long()

    def set_samples(self, samples: torch.Tensor):
        if self.shuffle:
            samples = samples.numpy()
            np.random.shuffle(samples)
            self.samples = torch.from_numpy(samples)
            # self.samples = samples[torch.randperm(len(samples))]
        else:
            self.samples = samples


class TrainingJobNegativeSamplingDistributed(TrainingJobNegativeSampling):
    def __init__(
        self,
        config,
        dataset,
        parent_job=None,
        model=None,
        optimizer=None,
        forward_only=False,
        parameter_client=None,
        init_for_load_only=False,
    ):
        self.parameter_client = parameter_client
        self.min_rank = get_min_rank(config)

        self.work_scheduler_client = SchedulerClient(config)
        (
            max_partition_entities,
            max_partition_relations,
        ) = self.work_scheduler_client.get_init_info()
        if model is None:
            model: KgeModel = KgeModel.create(
                config,
                dataset,
                parameter_client=parameter_client,
                max_partition_entities=max_partition_entities,
            )
        model.get_s_embedder().to_device()
        model.get_p_embedder().to_device()
        lapse_indexes = [
            torch.arange(dataset.num_entities(), dtype=torch.int),
            torch.arange(dataset.num_relations(), dtype=torch.int)
            + dataset.num_entities(),
        ]
        if optimizer is None:
            optimizer = KgeOptimizer.create(
                config,
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
            )
        # barrier to wait for loading of pretrained embeddings
        self.parameter_client.barrier()
        super().__init__(
            config,
            dataset,
            parent_job,
            model=model,
            optimizer=optimizer,
            forward_only=forward_only,
        )
        self.type_str = "negative_sampling"
        self.load_batch = self.config.get("job.distributed.load_batch")
        self.entity_localize = self.config.get("job.distributed.entity_localize")
        self.relation_localize = self.config.get("job.distributed.relation_localize")
        self.entity_async_write_back = self.config.get(
            "job.distributed.entity_async_write_back"
        )
        self.relation_async_write_back = self.config.get(
            "job.distributed.relation_async_write_back"
        )
        self.entity_sync_level = self.config.get("job.distributed.entity_sync_level")
        self.relation_sync_level = self.config.get(
            "job.distributed.relation_sync_level"
        )
        self.entity_pre_pull = self.config.get("job.distributed.entity_pre_pull")
        self.relation_pre_pull = self.config.get("job.distributed.relation_pre_pull")
        self.entity_mapper_tensors = deque()
        for i in range(self.config.get("train.num_workers") + 1):
            self.entity_mapper_tensors.append(
                torch.full((self.dataset.num_entities(),), -1, dtype=torch.long)
            )

        self._initialize_parameter_server(init_for_load_only=init_for_load_only)
        self.early_stop_hooks.append(lambda job: job.parameter_client.stop())

        if self.__class__ == TrainingJobNegativeSamplingDistributed:
            for f in Job.job_created_hooks:
                f(self)

    def _initialize_parameter_server(self, init_for_load_only=False):
        # initialize the parameter server
        #  each worker takes as many entities as it can fit, inits and pushes
        #  init work is distributed by the work scheduler
        if not init_for_load_only and not self.config.get(
            "lookup_embedder.pretrain.model_filename"
        ):
            # only the first worker initializes the relations
            if self.parameter_client.rank == self.min_rank:
                self.model.get_p_embedder().push_all()
            while True:
                init_entities = self.work_scheduler_client.get_init_work(
                    self.model.get_s_embedder().vocab_size
                )
                if init_entities is None:
                    break
                self.model.get_s_embedder().initialize(
                    self.model.get_s_embedder()._embeddings.weight.data
                )
                self.model.get_s_embedder()._normalize_embeddings()
                push_tensor = torch.cat(
                    (
                        self.model.get_s_embedder()
                        ._embeddings.weight.data[: len(init_entities)]
                        .cpu(),
                        self.model.get_s_embedder()
                        .optimizer_values[: len(init_entities)]
                        .cpu(),
                    ),
                    dim=1,
                )
                self.parameter_client.push(
                    init_entities + self.model.get_s_embedder().lapse_offset,
                    push_tensor.cpu(),
                )
        self.parameter_client.barrier()

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.dataloader_dataset = BatchDataset(
            self.dataset.split(self.train_split),
            batch_size=self.batch_size,
            shuffle=True,
        )
        mp_context = (
            torch.multiprocessing.get_context("fork")
            if self.config.get("train.num_workers") > 0
            else None
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataloader_dataset,
            collate_fn=self._get_collate_fun(),
            shuffle=False,  # shuffle needs to be False, since it is handled in the dataset object
            # batch_size=self.batch_size,  # batch size needs to be 1 since it is handled in the dataset object
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
            multiprocessing_context=mp_context,
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.split(self.train_split)[batch[0], :].long()

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            unique_time = -time.time()
            unique_entities = torch.unique(
                torch.cat(
                    (
                        triples[:, [S, O]].view(-1),
                        negative_samples[S].unique_samples(remove_dropped=False),
                        negative_samples[O].unique_samples(remove_dropped=False),
                    )
                )
            )
            unique_relations = torch.unique(
                torch.cat(
                    (
                        triples[:, [P]].view(-1),
                        negative_samples[P].unique_samples(remove_dropped=False),
                    )
                )
            )
            unique_time += time.time()

            # map ids to local ids
            if self.entity_sync_level == "partition":
                entity_mapper = self.model.get_s_embedder().global_to_local_mapper
            else:
                # entity_mapper = torch.full((self.dataset.num_entities(),), -1, dtype=torch.long)
                entity_mapper = self.entity_mapper_tensors.popleft()
                entity_mapper[unique_entities] = torch.arange(
                    len(unique_entities), dtype=torch.long
                )
            if self.relation_sync_level == "partition":
                relation_mapper = self.model.get_p_embedder().global_to_local_mapper
            else:
                relation_mapper = torch.full(
                    (self.dataset.num_relations(),), -1, dtype=torch.long
                )
                relation_mapper[unique_relations] = torch.arange(
                    len(unique_relations), dtype=torch.long
                )
            triples[:, S] = entity_mapper[triples[:, S]]
            triples[:, P] = relation_mapper[triples[:, P]]
            triples[:, O] = entity_mapper[triples[:, O]]
            negative_samples[S].map_samples(entity_mapper)
            negative_samples[P].map_samples(relation_mapper)
            negative_samples[O].map_samples(entity_mapper)

            # for debugging reset the entity mapper to -1
            # entity_mapper[:] = -1
            self.entity_mapper_tensors.append(entity_mapper)
            return {
                "triples": triples,
                "negative_samples": negative_samples,
                "unique_entities": unique_entities,
                "unique_relations": unique_relations,
                "unique_time": unique_time,
            }

        return collate

    def _prepare_batch_ahead(self, batches: deque):
        if self.entity_pre_pull > 1 or self.relation_pre_pull > 1:
            batches[0]["triples"] = batches[0]["triples"].to(self.device)
            for ns in batches[0]["negative_samples"]:
                ns.positive_triples = batches[0]["triples"]
            batches[0]["negative_samples"] = [
                ns.to(self.device) for ns in batches[0]["negative_samples"]
            ]
        if self.entity_sync_level == "batch" and self.entity_pre_pull > 0:
            self.model.get_s_embedder().pre_pull(batches[-1]["unique_entities"])
            self.model.get_s_embedder().pre_pulled_to_device()
        if self.relation_sync_level == "batch" and self.relation_pre_pull > 0:
            self.model.get_p_embedder().pre_pull(batches[-1]["unique_relations"])
            self.model.get_p_embedder().pre_pulled_to_device()

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        # result.cpu_gpu_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]
        # result.cpu_gpu_time += time.time()
        result.unique_time += batch["unique_time"]
        if self.config.get("job.distributed.load_batch"):
            if self.entity_sync_level == "batch":
                # result.unique_time -= time.time()
                unique_entities = batch["unique_entities"]
                # unique_entities = torch.unique(torch.cat((batch["triples"][:, [S,O]].view(-1), batch["negative_samples"][S].unique_samples(), batch["negative_samples"][O].unique_samples())))
                # result.unique_time += time.time()

                result.ps_wait_time -= time.time()
                if not self.entity_async_write_back:
                    for wait_value in self.optimizer.entity_async_wait_values:
                        self.parameter_client.wait(wait_value)
                    self.optimizer.entity_async_wait_values.clear()
                result.ps_wait_time += time.time()
                if self.entity_localize:
                    self.model.get_s_embedder().localize(unique_entities)
                result.pull_and_map_time -= time.time()
                (
                    entity_pull_time,
                    cpu_gpu_time,
                ) = self.model.get_s_embedder()._pull_embeddings(unique_entities)
                result.pull_and_map_time += time.time()
                result.entity_pull_time += entity_pull_time
                result.cpu_gpu_time += cpu_gpu_time
            if self.relation_sync_level == "batch":
                # result.unique_time -= time.time()
                unique_relations = batch["unique_relations"]
                # unique_relations = torch.unique(torch.cat((batch["triples"][:, [P]].view(-1), batch["negative_samples"][P].unique_samples())))
                # result.unique_time += time.time()
                result.ps_wait_time -= time.time()
                if not self.relation_async_write_back:
                    for wait_value in self.optimizer.relation_async_wait_values:
                        self.parameter_client.wait(wait_value)
                    self.optimizer.relation_async_wait_values.clear()
                result.ps_wait_time += time.time()
                if self.relation_localize:
                    self.model.get_p_embedder().localize(unique_relations)
                result.pull_and_map_time -= time.time()
                (
                    relation_pull_time,
                    cpu_gpu_time,
                ) = self.model.get_p_embedder()._pull_embeddings(unique_relations)
                result.pull_and_map_time += time.time()
                result.relation_pull_time += relation_pull_time
                result.cpu_gpu_time += cpu_gpu_time

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def handle_validation(self, metric_name):
        checkpoint_every = self.config.get("train.checkpoint.every")
        checkpoint_keep = self.config.get("train.checkpoint.keep")

        tmp_model = self.model.cpu()
        self.valid_job.model = tmp_model
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.parameter_client.barrier()
        if self.parameter_client.rank == self.min_rank:
            # move current small model to a tmp model
            # self.model = self.model.cpu()
            tmp_optimizer = self.optimizer
            # TODO: we also need to handle the learning rate scheduler somehow
            #  in the checkpoint

            # create a model for validation with entity embedder size
            #  batch_size x 2 + eval.chunk_size
            self.config.set(self.config.get("model") + ".create_eval", True)
            self.model = KgeModel.create(
                self.config, self.dataset, parameter_client=self.parameter_client
            )
            self.model.get_s_embedder().to_device(move_optim_data=False)
            self.model.get_p_embedder().to_device(move_optim_data=False)
            self.config.set(self.config.get("model") + ".create_eval", False)

            self.valid_job.model = self.model
            # validate and update learning rate
            if (
                    self.config.get("valid.every") > 0
                    and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self)
                self.model.meta["valid_trace_entry"] = trace_entry

                # metric-based scheduler step
                self.kge_lr_scheduler.step(trace_entry[metric_name])
            else:
                self.kge_lr_scheduler.step()

            # create a new complete model, to be able to store
            self.config.set(self.config.get("model") + ".create_complete", True)
            worker_folder = self.config.folder
            valid_folder = os.path.dirname(worker_folder)
            self.config.folder = valid_folder
            self.config.set("job.device", "cpu")
            self.model = KgeModel.create(
                self.config, self.dataset, parameter_client=self.parameter_client
            )
            self.model.get_s_embedder().pull_all()
            self.model.get_p_embedder().pull_all()
            self.optimizer = KgeOptimizer.create(
                self.config, self.model, parameter_client=self.parameter_client
            )
            self.optimizer.pull_all()
            self.config.set("job.device", self.device)
            # self.model = self.model.to(self.device)
            # we need to move some mappers separately to device
            # self.model.get_s_embedder().to_device(move_optim_data=False)
            # self.model.get_p_embedder().to_device(move_optim_data=False)

            # create checkpoint and delete old one, if necessary
            self.save(self.config.checkpoint_file(self.epoch))
            if (
                    len(self.valid_trace) > 0
                    and self.valid_trace[-1]["epoch"] == self.epoch
            ):
                best_index = max(
                    range(len(self.valid_trace)),
                    key=lambda index: self.valid_trace[index][metric_name],
                )
                if best_index == len(self.valid_trace) - 1:
                    self.save(self.config.checkpoint_file("best"))
            if self.epoch > 1:
                delete_checkpoint_epoch = -1
                if checkpoint_every == 0:
                    # do not keep any old checkpoints
                    delete_checkpoint_epoch = self.epoch - 1
                # in the distributed setup we only save checkpoints when we evaluate
                #  since it is expensive to create the complete model
                # therefore checkpoint every does not work
                # elif (self.epoch - 1) % checkpoint_every != 0:
                #     # delete checkpoints that are not in the checkpoint.every schedule
                #     delete_checkpoint_epoch = self.epoch - 1
                elif checkpoint_keep > 0:
                    # keep a maximum number of checkpoint_keep checkpoints
                    # since in distributed setup we only create checkpoints when
                    #  we evaluate, checkpoint_keep needs to refer to valid.every
                    # delete_checkpoint_epoch = (
                    #     self.epoch - checkpoint_every * checkpoint_keep
                    # )
                    delete_checkpoint_epoch = (
                            self.epoch - self.config.get(
                        "valid.every") * checkpoint_keep
                    )
                if delete_checkpoint_epoch > 0:
                    if os.path.exists(
                            self.config.checkpoint_file(delete_checkpoint_epoch)
                    ):
                        self.config.log(
                            "Removing old checkpoint {}...".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )
                        os.remove(
                            self.config.checkpoint_file(delete_checkpoint_epoch)
                        )
                    else:
                        self.config.log(
                            "Could not delete old checkpoint {}, does not exits.".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )
            self.config.set(self.config.get("model") + ".create_complete", False)
            self.config.folder = worker_folder
            # self.model = self.model.cpu()
            del self.optimizer
            del self.model
            del self.valid_job.model
            gc.collect()
            torch.cuda.empty_cache()
            self.optimizer = tmp_optimizer
            self.model = tmp_model.to(self.device)
            del tmp_optimizer
        else:
            self.kge_lr_scheduler.step()
        self.parameter_client.barrier()
        self.model = tmp_model.to(self.device)
        del tmp_model
        gc.collect()

    def handle_running_checkpoint(self, checkpoint_every, checkpoint_keep):
        # do nothing since we are handling this in handle validation currently
        pass
