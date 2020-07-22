from kge import Config, Configurable
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
from kge.util.dist_sgd import DistSGD
from kge.util.dist_adagrad import DistAdagrad


class KgeOptimizer:
    """ Wraps torch optimizers """

    @staticmethod
    def create(config, model, parameter_client=None, lapse_indexes=None):
        """ Factory method for optimizer creation """
        if config.get("train.optimizer") == "dist_sgd":
            optimizer = DistSGD(
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
                **config.get("train.optimizer_args"),
            )
            return optimizer
        if config.get("train.optimizer") == "dist_adagrad":
            optimizer = DistAdagrad(
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
                sync_levels=[
                    config.get("job.distributed.entity_sync_level"),
                    config.get("job.distributed.relation_sync_level"),
                ],
                **config.get("train.optimizer_args"),
            )
            return optimizer
        else:
            try:
                optimizer = getattr(torch.optim, config.get("train.optimizer"))
                return optimizer(
                    [p for p in model.parameters() if p.requires_grad],
                    **config.get("train.optimizer_args"),
                )
            except AttributeError:
                # perhaps TODO: try class with specified name -> extensibility
                raise ValueError(
                    f"Could not create optimizer {config.get('train.optimizer')}. "
                    f"Please specify an optimizer provided in torch.optim"
                )


class KgeLRScheduler(Configurable):
    """ Wraps torch learning rate (LR) schedulers """

    def __init__(self, config: Config, optimizer):
        super().__init__(config)
        name = config.get("train.lr_scheduler")
        args = config.get("train.lr_scheduler_args")
        self._lr_scheduler: _LRScheduler = None
        if name != "":
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
                    optimizer, **args
                )
            except Exception as e:
                raise ValueError(
                    (
                        "Invalid LR scheduler {} or scheduler arguments {}. "
                        "Error: {}"
                    ).format(name, args, e)
                )

        self._metric_based = name in ["ReduceLROnPlateau"]

    def step(self, metric=None):
        if self._lr_scheduler is None:
            return
        if self._metric_based:
            if metric is not None:
                # metric is set only after validation has been performed, so here we
                # step
                self._lr_scheduler.step(metrics=metric)
        else:
            # otherwise, step after every epoch
            self._lr_scheduler.step()

    def state_dict(self):
        if self._lr_scheduler is None:
            return dict()
        else:
            return self._lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        if self._lr_scheduler is None:
            pass
        else:
            self._lr_scheduler.load_state_dict(state_dict)
