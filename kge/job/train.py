import itertools
import os
import math
import time
import traceback
import gc
from collections import defaultdict, deque

from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from kge.util.io import load_checkpoint
from kge.job.trace import format_trace_entry

# fixme: for some reason python from console cries about circular imports if loaded
#  from init. But directly it works (partially initialized model)
from kge.distributed.work_scheduler import SchedulerClient
from kge.distributed.parameter_client import KgeParameterClient
from kge.distributed.misc import MIN_RANK

# from kge.distributed import KgeParameterClient, SchedulerClient
from typing import Any, Callable, Dict, List, Optional
import kge.job.util
from kge.util.metric import Metric

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


def _generate_worker_init_fn(config):
    "Initialize workers of a DataLoader"
    use_fixed_seed = config.get("random_seed.numpy") >= 0

    def worker_init_fn(worker_num):
        # ensure that NumPy uses different seeds at each worker
        if use_fixed_seed:
            # reseed based on current seed (same for all workers) and worker number
            # (different)
            base_seed = np.random.randint(2 ** 32 - 1)
            np.random.seed(base_seed + worker_num)
        else:
            # reseed fresh
            np.random.seed()

    return worker_init_fn


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.samples = list(range(num_samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx
    #    return self.samples[idx]

    def set_samples(self, samples):
        self.samples = samples


class TrainingJob(TrainingOrEvaluationJob):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
        parameter_client: Optional[KgeParameterClient] = None,
        init_for_load_only=False,
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        self.parameter_client = parameter_client
        self.entity_sync_level = self.config.get("job.distributed.entity_sync_level")
        self.relation_sync_level = self.config.get(
            "job.distributed.relation_sync_level"
        )
        # here we create one large model to init lapse and remove it afterwards
        if not init_for_load_only:
            self.parameter_client.barrier()
            if self.parameter_client.rank == MIN_RANK:
                self.config.set(self.config.get("model") + ".create_complete", True)
                init_model = KgeModel.create(
                    self.config, self.dataset, parameter_client=self.parameter_client
                )
                init_model.get_s_embedder().push_all()
                init_model.get_p_embedder().push_all()
                del init_model
                gc.collect()
                self.config.set(self.config.get("model") + ".create_complete", False)
            self.parameter_client.barrier()

        self.work_scheduler_client = SchedulerClient()
        (
            max_partition_entities,
            max_partition_relations,
        ) = self.work_scheduler_client.get_init_info()
        if model is None:
            self.model: KgeModel = KgeModel.create(
                config,
                dataset,
                parameter_client=parameter_client,
                max_partition_entities=max_partition_entities,
            )
        else:
            self.model: KgeModel = model
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.batch_size: int = config.get("train.batch_size")
        self._subbatch_auto_tune: bool = config.get("train.subbatch_auto_tune")
        self._max_subbatch_size: int = config.get("train.subbatch_size")
        self.device: str = self.config.get("job.device")
        self.train_split = config.get("train.split")

        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.is_forward_only = forward_only

        if not self.is_forward_only:
            self.model.train()
            lapse_indexes = [
                torch.arange(dataset.num_entities(), dtype=torch.int),
                torch.arange(dataset.num_relations(), dtype=torch.int)
                + dataset.num_entities(),
                ]
            self.model.get_s_embedder().to_device()
            self.model.get_p_embedder().to_device()
            self.optimizer = KgeOptimizer.create(
                config,
                self.model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
            )
            self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)

            self.valid_trace: List[Dict[str, Any]] = []
            valid_conf = config.clone()
            valid_conf.set("job.type", "eval")
            if self.config.get("valid.split") != "":
                valid_conf.set("eval.split", self.config.get("valid.split"))
            valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
            self.valid_job = EvaluationJob.create(
                valid_conf, dataset, parent_job=self, model=self.model
            )
        self.entity_pre_pull = self.config.get("job.distributed.entity_pre_pull")
        self.relation_pre_pull = self.config.get("job.distributed.relation_pre_pull")

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        # Hooks run after validation. The corresponding valid trace entry can be found
        # in self.valid_trace[-1] Signature: job
        self.post_valid_hooks: List[Callable[[Job], Any]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config,
        dataset: Dataset,
        parent_job: Job = None,
        model=None,
        forward_only=False,
        parameter_client=None,
        init_for_load_only=False,
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "KvsAll":
            return TrainingJobKvsAll(
                config,
                dataset,
                parent_job,
                model=model,
                forward_only=forward_only,
                parameter_client=parameter_client,
            )
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(
                config,
                dataset,
                parent_job,
                model=model,
                forward_only=forward_only,
                parameter_client=parameter_client,
                init_for_load_only=init_for_load_only,
            )
        elif config.get("train.type") == "1vsAll":
            return TrainingJob1vsAll(
                config,
                dataset,
                parent_job,
                model=model,
                forward_only=forward_only,
                parameter_client=parameter_client,
            )
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def _run(self) -> None:
        """Start/resume the training job and run to completion."""

        if self.is_forward_only:
            raise Exception(
                f"{self.__class__.__name__} was initialized for forward only. You can only call run_epoch()"
            )

        self.config.log("Starting training...")
        checkpoint_every = self.config.get("train.checkpoint.every")
        checkpoint_keep = self.config.get("train.checkpoint.keep")
        metric_name = self.config.get("valid.metric")
        patience = self.config.get("valid.early_stopping.patience")
        while True:
            # checking for model improvement according to metric_name
            # and do early stopping and keep the best checkpoint
            if (
                len(self.valid_trace) > 0
                and self.valid_trace[-1]["epoch"] == self.epoch
            ):
                best_index = Metric(self).best_index(
                    list(map(lambda trace: trace[metric_name], self.valid_trace))
                )
                # we are now saving the best checkpoint directly after validating
                # otherwise we would have to create the complete model again here
                # if best_index == len(self.valid_trace) - 1:
                #     self.save(self.config.checkpoint_file("best"))
                if (
                    patience > 0
                    and len(self.valid_trace) > patience
                    and best_index < len(self.valid_trace) - patience
                ):
                    self.config.log(
                        "Stopping early ({} did not improve over best result ".format(
                            metric_name
                        )
                        + "in the last {} validation runs).".format(patience)
                    )
                    self.parameter_client.stop()
                    # break
                elif self.epoch > self.config.get(
                    "valid.early_stopping.threshold.epochs"
                ):
                    achieved = self.valid_trace[best_index][metric_name]
                    target = self.config.get(
                        "valid.early_stopping.threshold.metric_value"
                    )
                    if Metric(self).better(target, achieved):
                        self.config.log(
                            "Stopping early ({} did not achieve threshold after {} epochs".format(
                                metric_name, self.epoch
                            )
                        )
                        self.parameter_client.stop()
                        # self.work_scheduler_client.shutdown()
                        # break

            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break

            self.parameter_client.barrier()
            if self.parameter_client.is_stopped():
                self.config.log(f"Shutting down Trainer {self.parameter_client.rank}")
                break

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            print("done worker: ", self.parameter_client.rank)
            if self.config.get("valid.every") > 0 and self.epoch % self.config.get("valid.every") == 0:
                tmp_model = self.model.cpu()
                self.valid_job.model = tmp_model
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                self.parameter_client.barrier()
                if self.parameter_client.rank == MIN_RANK:
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
                                    self.epoch - self.config.get("valid.every") * checkpoint_keep
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
            else:
                self.kge_lr_scheduler.step()

        self.trace(event="train_completed")

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        checkpoint = self.save_to({})
        torch.save(
            checkpoint,
            filename,
        )

    def save_to(self, checkpoint: Dict) -> Dict:
        """Adds trainjob specific information to the checkpoint"""
        train_checkpoint = {
            "type": "train",
            "epoch": self.epoch,
            "valid_trace": self.valid_trace,
            "model": self.model.save(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
            "job_id": self.job_id,
        }
        train_checkpoint = self.config.save_to(train_checkpoint)
        checkpoint.update(train_checkpoint)
        return checkpoint

    def _load(self, checkpoint: Dict) -> str:
        if checkpoint["type"] != "train":
            raise ValueError("Training can only be continued on trained checkpoints")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint:
            # new format
            self.kge_lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()
        self.resumed_from_job_id = checkpoint.get("job_id")
        self.trace(
            event="job_resumed",
            epoch=self.epoch,
            checkpoint_file=checkpoint["file"],
        )
        self.config.log(
            "Resuming training from {} of job {}".format(
                checkpoint["file"], self.resumed_from_job_id
            )
        )

    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        trace_entry = None
        while True:
            # variables that record various statitics
            sum_loss = 0.0
            sum_penalty = 0.0
            sum_penalties = defaultdict(lambda: 0.0)
            epoch_time = -time.time()
            prepare_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optimizer_time = 0.0
            unique_time = 0.0
            pull_and_map_time = 0.0
            entity_pull_time = 0.0
            relation_pull_time = 0.0
            pre_pull_time = 0.0
            cpu_gpu_time = 0.0
            ps_wait_time = 0.0
            ps_set_time = 0.0
            scheduler_time = -time.time()

            # load new work package
            work, work_entities, work_relations = self.work_scheduler_client.get_work()
            if work is None:
                break
            self.dataloader_dataset.set_samples(work)
            if self.entity_sync_level == "partition":
                if work_entities is not None:
                    entity_pull_time -= time.time()
                    self.model.get_s_embedder()._pull_embeddings(work_entities)
                    self.model.get_s_embedder().global_to_local_mapper[work_entities] = torch.arange(len(work_entities), dtype=torch.long, device="cpu")
                    entity_pull_time += time.time()
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing entities on a partition level"
                    )
            if self.relation_sync_level == "partition":
                if work_relations is not None:
                    relation_pull_time -= time.time()
                    self.model.get_p_embedder()._pull_embeddings(work_relations)
                    self.model.get_p_embedder().global_to_local_mapper[work_relations] = torch.arange(len(work_relations), dtype=torch.long, device="cpu")
                    relation_pull_time += time.time()
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing relations on a partition level"
                    )

            if (
                work_entities is not None
                and self.config.get("negative_sampling.sampling_type") == "pooled"
            ):
                self._sampler.set_pool(work_entities, S)
                self._sampler.set_pool(work_entities, O)
            scheduler_time += time.time()

            # process each batch
            pre_load_batches = deque()
            batch = None
            epoch_done = False
            iter_dataloader = iter(self.loader)
            batch_index = 0
            num_prepulls = max(self.entity_pre_pull, self.relation_pre_pull, 1)
            #for batch_index, batch in enumerate(self.loader):
            while not epoch_done:
                try:
                    if batch is None and len(pre_load_batches) < num_prepulls:
                        pre_load_batches.append(next(iter_dataloader))
                        prepare_time -= time.time()
                        pre_pull_time -= time.time()
                        self._prepare_batch_ahead(pre_load_batches)
                        pre_pull_time += time.time()
                        prepare_time += time.time()
                        continue
                    else:
                        batch = pre_load_batches.popleft()
                    pre_load_batches.append(next(iter_dataloader))
                    prepare_time -= time.time()
                    pre_pull_time -= time.time()
                    self._prepare_batch_ahead(pre_load_batches)
                    pre_pull_time += time.time()
                    prepare_time += time.time()
                except StopIteration:
                    if len(pre_load_batches) == 0:
                        epoch_done = True


                # create initial batch trace (yet incomplete)
                self.current_trace["batch"] = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "split": self.train_split,
                    "batch": batch_index,
                    "batches": len(self.loader),
                    }
                if not self.is_forward_only:
                    self.current_trace["batch"].update(
                        lr=[group["lr"] for group in self.optimizer.param_groups],
                    )

                # run the pre-batch hooks (may update the trace)
                for f in self.pre_batch_hooks:
                    f(self)

                # process batch (preprocessing + forward pass + backward pass on loss)
                done = False
                while not done:
                    try:
                        # try running the batch
                        if not self.is_forward_only:
                            self.optimizer.zero_grad()
                        batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                            batch_index, batch)
                        done = True
                    except RuntimeError as e:
                        # is it a CUDA OOM exception and are we allowed to reduce the
                        # subbatch size on such an error? if not, raise the exception again
                        if (
                            "CUDA out of memory" not in str(e)
                            or not self._subbatch_auto_tune
                        ):
                            raise e

                        # try rerunning with smaller subbatch size
                        tb = traceback.format_exc()
                        self.config.log(tb)
                        self.config.log(
                            "Caught OOM exception when running a batch; "
                            "trying to reduce the subbatch size..."
                        )

                        if self._max_subbatch_size <= 0:
                            self._max_subbatch_size = self.batch_size
                        if self._max_subbatch_size <= 1:
                            self.config.log(
                                "Cannot reduce subbatch size "
                                f"(current value: {self._max_subbatch_size})"
                            )
                            raise e  # cannot reduce further

                        self._max_subbatch_size //= 2
                        self.config.set(
                            "train.subbatch_size", self._max_subbatch_size, log=True
                        )
                sum_loss += batch_result.avg_loss * batch_result.size

                # determine penalty terms (forward pass)
                batch_forward_time = batch_result.forward_time - time.time()
                penalties_torch = self.model.penalty(
                    epoch=self.epoch,
                    batch_index=batch_index,
                    num_batches=len(self.loader),
                    batch=batch,
                )
                batch_forward_time += time.time()

                # backward pass on penalties
                batch_backward_time = batch_result.backward_time - time.time()
                penalty = 0.0
                for index, (penalty_key, penalty_value_torch) in enumerate(
                    penalties_torch
                ):
                    if not self.is_forward_only:
                        penalty_value_torch.backward()
                    penalty += penalty_value_torch.item()
                    sum_penalties[penalty_key] += penalty_value_torch.item()
                sum_penalty += penalty
                batch_backward_time += time.time()

                # determine full cost
                cost_value = batch_result.avg_loss + penalty

                # abort on nan
                if self.abort_on_nan and math.isnan(cost_value):
                    raise FloatingPointError("Cost became nan, aborting training job")

                # TODO # visualize graph
                # if (
                #     self.epoch == 1
                #     and batch_index == 0
                #     and self.config.get("train.visualize_graph")
                # ):
                #     from torchviz import make_dot

                #     f = os.path.join(self.config.folder, "cost_value")
                #     graph = make_dot(cost_value, params=dict(self.model.named_parameters()))
                #     graph.save(f"{f}.gv")
                #     graph.render(f)  # needs graphviz installed
                #     self.config.log("Exported compute graph to " + f + ".{gv,pdf}")

                # print memory stats
                if self.epoch == 1 and batch_index == 0:
                    if self.device.startswith("cuda"):
                        self.config.log(
                            "CUDA memory after first batch: allocated={:14,} "
                            "reserved={:14,} max_allocated={:14,}".format(
                                torch.cuda.memory_allocated(self.device),
                                torch.cuda.memory_reserved(self.device),
                                torch.cuda.max_memory_allocated(self.device),
                            )
                        )

                # update parameters
                batch_optimizer_time = -time.time()
                if not self.is_forward_only:
                    self.optimizer.step()
                batch_optimizer_time += time.time()

                if self.entity_sync_level == "batch":
                    self.model.get_s_embedder().push_back()
                if self.relation_sync_level == "batch":
                    self.model.get_p_embedder().push_back()

                # update batch trace with the results
                self.current_trace["batch"].update(
                    {
                        "size": batch_result.size,
                        "avg_loss": batch_result.avg_loss,
                        #"penalties": [p.item() for k, p in penalties_torch],
                        "penalty": penalty,
                        "cost": cost_value,
                        "prepare_time": batch_result.prepare_time,
                        "forward_time": batch_forward_time,
                        "backward_time": batch_backward_time,
                        "optimizer_time": batch_optimizer_time,
                        "event": "batch_completed",
                    }
                )

                # run the post-batch hooks (may modify the trace)
                for f in self.post_batch_hooks:
                    f(self)

                # output, then clear trace
                if self.trace_batch:
                        self.trace(**self.current_trace["batch"])
                self.current_trace["batch"] = None

                # print console feedback
                self.config.print(
                    (
                        "\r"  # go back
                        + "{}  batch{: "
                        + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                        + "d}/{}"
                        + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                        + "\033[K"  # clear to right
                    ).format(
                        self.config.log_prefix,
                        batch_index,
                        len(self.loader) - 1,
                        batch_result.avg_loss,
                        penalty,
                        cost_value,
                        batch_result.prepare_time
                        + batch_forward_time
                        + batch_backward_time
                        + batch_optimizer_time,
                    ),
                    end="",
                    flush=True,
                )

                # update epoch times
                prepare_time += batch_result.prepare_time
                forward_time += batch_forward_time
                backward_time += batch_backward_time
                optimizer_time += batch_optimizer_time
                pull_and_map_time += batch_result.pull_and_map_time
                entity_pull_time += batch_result.entity_pull_time
                relation_pull_time += batch_result.relation_pull_time
                unique_time += batch_result.unique_time
                cpu_gpu_time += batch_result.cpu_gpu_time
                ps_wait_time += batch_result.ps_wait_time

                batch_index += 1

            # all done; now trace and log
            epoch_time += time.time()
            self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

            other_time = (
                epoch_time
                - prepare_time
                - forward_time
                - backward_time
                - optimizer_time
                - scheduler_time
            )

            print("work done", self.parameter_client.rank)
            if self.entity_sync_level == "partition":
                ps_set_time -= time.time()
                self.model.get_s_embedder().set_embeddings()
                ps_set_time += time.time()
                self.model.get_s_embedder().global_to_local_mapper[:] = -1
                self.model.get_s_embedder().push_back()
            if self.relation_sync_level == "partition":
                ps_set_time -= time.time()
                self.model.get_p_embedder().set_embeddings()
                ps_set_time += time.time()
                self.model.get_p_embedder().global_to_local_mapper[:] = -1
                self.model.get_p_embedder().push_back()
            self.work_scheduler_client.work_done()

            # add results to trace entry
            self.current_trace["epoch"].update(
                dict(
                    avg_loss=sum_loss / self.num_examples,
                    avg_penalty=sum_penalty / len(self.loader),
                    avg_penalties={
                        k: p / len(self.loader) for k, p in sum_penalties.items()

                    },
                    avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
                    epoch_time=epoch_time,
                    prepare_time=prepare_time,
                    ps_wait_time=ps_wait_time,
                    unique_time=unique_time,
                    pull_and_map_time=pull_and_map_time,
                    pre_pull_time=pre_pull_time,
                    entity_pull_time=entity_pull_time,
                    relation_pull_time=relation_pull_time,
                    ps_set_time=ps_set_time,
                    cpu_gpu_time=cpu_gpu_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    optimizer_time=optimizer_time,
                    scheduler_time=scheduler_time,
                    other_time=other_time,
                    embedding_mapping_time=self.model.get_s_embedder().mapping_time + self.model.get_p_embedder().mapping_time,
                    event="epoch_completed",
                )
            )
            self.model.get_p_embedder().mapping_time = 0.0
            self.model.get_s_embedder().mapping_time = 0.0


            # run hooks (may modify trace)
            for f in self.post_epoch_hooks:
                f(self)

            # output the trace, then clear it
            trace_entry = self.trace(
                **self.current_trace["epoch"], echo=False, echo_prefix="  ", log=True
            )
        self.config.log(format_trace_entry("train_epoch", trace_entry, self.config))
        self.current_trace["epoch"] = None
        return trace_entry

    def _prepare_batch_ahead(self, batches: deque):
        pass

    def _prepare(self):
        """Prepare this job for running.

        Sets (at least) the `loader`, `num_examples`, and `type_str` attributes of this
        job to a data loader, number of examples per epoch, and a name for the trainer,
        repectively.

        Guaranteed to be called exactly once before running the first epoch.

        """
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

    @dataclass
    class _ProcessBatchResult:
        """Result of running forward+backward pass on a batch."""

        avg_loss: float = 0.0
        size: int = 0
        prepare_time: float = 0.0
        forward_time: float = 0.0
        backward_time: float = 0.0
        pull_and_map_time: float = 0.0
        entity_pull_time: float = 0.0
        relation_pull_time: float = 0.0
        unique_time: float = 0.0
        cpu_gpu_time: float = 0.0
        ps_wait_time: float = 0.0

    def _process_batch(self, batch_index, batch) -> _ProcessBatchResult:
        "Breaks a batch into subbatches and processes them in turn."
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._process_subbatch(batch_index, batch, subbatch_slice, result)

        return result

    def _prepare_batch(self, batch_index, batch, result: _ProcessBatchResult):
        """Prepare the given batch for processing and determine the batch size.

        batch size must be written into result.size.
        """
        raise NotImplementedError

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: _ProcessBatchResult,
    ):
        """Run forward and backward pass on the given subbatch.

        Also update result.

        """
        raise NotImplementedError


class TrainingJobKvsAll(TrainingJob):
    """Train with examples consisting of a query and its answers.

    Terminology:
    - Query type: which queries to ask (sp_, s_o, and/or _po), can be configured via
      configuration key `KvsAll.query_type` (which see)
    - Query: a particular query, e.g., (John,marriedTo) of type sp_
    - Labels: list of true answers of a query (e.g., [Jane])
    - Example: a query + its labels, e.g., (John,marriedTo), [Jane]
    """

    from kge.indexing import KvsAllIndex

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False, parameter_client=None,
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only, parameter_client=parameter_client,
        )
        self.label_smoothing = config.check_range(
            "KvsAll.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "KvsAll"

        if self.__class__ == TrainingJobKvsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # determine enabled query types
        self.query_types = [
            key
            for key, enabled in self.config.get("KvsAll.query_types").items()
            if enabled
        ]

        # corresponding indexes
        self.query_indexes: List[KvsAllIndex] = []

        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        self.query_last_example = []

        # construct relevant data structures
        self.num_examples = 0
        for query_type in self.query_types:
            index_type = (
                "sp_to_o"
                if query_type == "sp_"
                else ("so_to_p" if query_type == "s_o" else "po_to_s")
            )
            index = self.dataset.index(f"{self.train_split}_{index_type}")
            self.query_indexes.append(index)
            self.num_examples += len(index)
            self.query_last_example.append(self.num_examples)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a dictionary of:

            - queries: nx2 tensor, row = query (sp, po, or so indexes)
            - label_coords: for each query, position of true answers (an Nx2 tensor,
              first columns holds query index, second colum holds index of label)
            - query_type_indexes (vector of size n holding the query type of each query)
            - triples (all true triples in the batch; e.g., needed for weighted
              penalties)

            """

            # count how many labels we have across the entire batch
            num_ones = 0
            for example_index in batch:
                start = 0
                for query_type_index in range(len(self.query_types)):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        num_ones += self.query_indexes[query_type_index]._values_offset[
                            example_index + 1
                        ]
                        num_ones -= self.query_indexes[query_type_index]._values_offset[
                            example_index
                        ]
                        break
                    start = end

            # now create the batch elements
            queries_batch = torch.zeros([len(batch), 2], dtype=torch.long)
            query_type_indexes_batch = torch.zeros([len(batch)], dtype=torch.long)
            label_coords_batch = torch.zeros([num_ones, 2], dtype=torch.int)
            triples_batch = torch.zeros([num_ones, 3], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                start = 0
                for query_type_index, query_type in enumerate(self.query_types):
                    end = self.query_last_example[query_type_index]
                    if example_index < end:
                        example_index -= start
                        query_type_indexes_batch[batch_index] = query_type_index
                        queries = self.query_indexes[query_type_index]._keys
                        label_offsets = self.query_indexes[
                            query_type_index
                        ]._values_offset
                        labels = self.query_indexes[query_type_index]._values
                        if query_type == "sp_":
                            query_col_1, query_col_2, target_col = S, P, O
                        elif query_type == "s_o":
                            query_col_1, target_col, query_col_2 = S, P, O
                        else:
                            target_col, query_col_1, query_col_2 = S, P, O
                        break
                    start = end

                queries_batch[
                    batch_index,
                ] = queries[example_index]
                start = label_offsets[example_index]
                end = label_offsets[example_index + 1]
                size = end - start
                label_coords_batch[
                    current_index : (current_index + size), 0
                ] = batch_index
                label_coords_batch[current_index : (current_index + size), 1] = labels[
                    start:end
                ]
                triples_batch[
                    current_index : (current_index + size), query_col_1
                ] = queries[example_index][0]
                triples_batch[
                    current_index : (current_index + size), query_col_2
                ] = queries[example_index][1]
                triples_batch[
                    current_index : (current_index + size), target_col
                ] = labels[start:end]
                current_index += size

            # all done
            return {
                "queries": queries_batch,
                "label_coords": label_coords_batch,
                "query_type_indexes": query_type_indexes_batch,
                "triples": triples_batch,
            }

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move labels to GPU for entire batch (else somewhat costly, but this should be
        # reasonably small)
        result.prepare_time -= time.time()
        batch["label_coords"] = batch["label_coords"].to(self.device)
        result.size = len(batch["queries"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        queries_subbatch = batch["queries"][subbatch_slice].to(self.device)
        batch_size = len(batch["queries"])
        subbatch_size = len(queries_subbatch)
        label_coords_batch = batch["label_coords"]
        query_type_indexes_subbatch = batch["query_type_indexes"][subbatch_slice]

        # in this method, example refers to the index of an example in the batch, i.e.,
        # it takes values in 0,1,...,batch_size-1
        examples_for_query_type = {}
        for query_type_index, query_type in enumerate(self.query_types):
            examples_for_query_type[query_type] = (
                (query_type_indexes_subbatch == query_type_index)
                .nonzero(as_tuple=False)
                .to(self.device)
                .view(-1)
            )

        labels_subbatch = kge.job.util.coord_to_sparse_tensor(
            subbatch_size,
            max(self.dataset.num_entities(), self.dataset.num_relations()),
            label_coords_batch,
            self.device,
            row_slice=subbatch_slice,
        ).to_dense()
        labels_for_query_type = {}
        for query_type, examples in examples_for_query_type.items():
            if query_type == "s_o":
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_relations()
                ]
            else:
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_entities()
                ]

        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            for query_type, labels in labels_for_query_type.items():
                if query_type != "s_o":  # entity targets only for now
                    labels_for_query_type[query_type] = (
                        1.0 - self.label_smoothing
                    ) * labels + 1.0 / labels.size(1)

        result.prepare_time += time.time()

        # forward/backward pass (sp)
        for query_type, examples in examples_for_query_type.items():
            if len(examples) > 0:
                result.forward_time -= time.time()
                if query_type == "sp_":
                    scores = self.model.score_sp(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.model.score_so(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                else:
                    scores = self.model.score_po(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                # note: average on batch_size, not on subbatch_size
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type]) / batch_size
                )
                result.avg_loss += loss_value.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    loss_value.backward()
                result.backward_time += time.time()


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(
        self,
        config,
        dataset,
        parent_job=None,
        model=None,
        forward_only=False,
        parameter_client=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config,
            dataset,
            parent_job,
            model=model,
            forward_only=forward_only,
            parameter_client=parameter_client,
            init_for_load_only=init_for_load_only,
        )
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self._implementation = self.config.check(
            "negative_sampling.implementation",
            ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            elif max_nr_of_negs > 30:
                self._implementation = "batch"

        config.log(
            "Initializing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )
        self.type_str = "negative_sampling"
        self.load_batch = self.config.get("job.distributed.load_batch")
        self.entity_localize = self.config.get("job.distributed.entity_localize")
        self.relation_localize = self.config.get("job.distributed.relation_localize")
        self.entity_async_write_back = self.config.get("job.distributed.entity_async_write_back")
        self.relation_async_write_back = self.config.get("job.distributed.relation_async_write_back")

        if self.__class__ == TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.dataloader_dataset = NumberDataset(self.num_examples)
        mp_context = torch.multiprocessing.get_context("fork") if self.config.get("train.num_workers") > 0 else None
        self.loader = torch.utils.data.DataLoader(
            # range(self.num_examples),
            self.dataloader_dataset,
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
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

            triples = self.dataset.split(self.train_split)[self.dataloader_dataset.samples[batch], :].long()
            # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
            # labels[:, 0] = 1
            # labels = labels.view(-1)

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            unique_time = -time.time()
            unique_entities = torch.unique(torch.cat((triples[:, [S, O]].view(-1),
                                                      negative_samples[S].unique_samples(remove_dropped=False),
                                                      negative_samples[
                                                          O].unique_samples(remove_dropped=False))))
            unique_relations = torch.unique(torch.cat((
                triples[:, [P]].view(-1), negative_samples[P].unique_samples(remove_dropped=False))))
            unique_time += time.time()

            # map ids to local ids
            if self.entity_sync_level == "partition":
                entity_mapper = self.model.get_s_embedder().global_to_local_mapper
            else:
                entity_mapper = torch.full((self.dataset.num_entities(),), -1, dtype=torch.long)
                entity_mapper[unique_entities] = torch.arange(len(unique_entities), dtype=torch.long)
            if self.relation_sync_level == "partition":
                relation_mapper = self.model.get_p_embedder().global_to_local_mapper
            else:
                relation_mapper = torch.full((self.dataset.num_relations(),), -1, dtype=torch.long)
                relation_mapper[unique_relations] = torch.arange(len(unique_relations), dtype=torch.long)
            triples[:, S] = entity_mapper[triples[:, S]]
            triples[:, P] = relation_mapper[triples[:, P]]
            triples[:, O] = entity_mapper[triples[:, O]]
            negative_samples[S].map_samples(entity_mapper)
            negative_samples[P].map_samples(relation_mapper)
            negative_samples[O].map_samples(entity_mapper)
            return {"triples": triples, "negative_samples": negative_samples, "unique_entities": unique_entities, "unique_relations": unique_relations, "unique_time": unique_time}

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
        #result.cpu_gpu_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]
        #result.cpu_gpu_time += time.time()
        result.unique_time += batch["unique_time"]
        if self.config.get("job.distributed.load_batch"):
            if self.entity_sync_level == "batch":
                #result.unique_time -= time.time()
                unique_entities = batch["unique_entities"]
                # unique_entities = torch.unique(torch.cat((batch["triples"][:, [S,O]].view(-1), batch["negative_samples"][S].unique_samples(), batch["negative_samples"][O].unique_samples())))
                #result.unique_time += time.time()

                result.ps_wait_time -= time.time()
                if not self.entity_async_write_back:
                    for wait_value in self.optimizer.entity_async_wait_values:
                          self.parameter_client.wait(wait_value)
                    self.optimizer.entity_async_wait_values.clear()
                result.ps_wait_time += time.time()
                if self.entity_localize:
                    self.model.get_s_embedder().localize(unique_entities)
                result.pull_and_map_time -= time.time()
                entity_pull_time, cpu_gpu_time = self.model.get_s_embedder()._pull_embeddings(unique_entities)
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
                relation_pull_time, cpu_gpu_time = self.model.get_p_embedder()._pull_embeddings(unique_relations)
                result.pull_and_map_time += time.time()
                result.relation_pull_time += relation_pull_time
                result.cpu_gpu_time += cpu_gpu_time

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        batch_size = len(batch["triples"])
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S],
                triples[:, P],
                triples[:, O],
                direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()


class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False, parameter_client=None
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only, parameter_client=parameter_client
        )
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {
                "triples": self.dataset.split(self.train_split)[batch, :].long()
            },
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        result.size = len(batch["triples"])

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        batch_size = len(triples)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time -= time.time()
        scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        result.avg_loss += loss_value_sp.item()
        result.forward_time += time.time()
        result.backward_time = -time.time()
        if not self.is_forward_only:
            loss_value_sp.backward()
        result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        result.avg_loss += loss_value_po.item()
        result.forward_time += time.time()
        result.backward_time -= time.time()
        if not self.is_forward_only:
            loss_value_po.backward()
        result.backward_time += time.time()
