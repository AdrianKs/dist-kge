import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from copy import deepcopy

# TODO: pull optimizer values from lapse before computing anything
#  support lapse sparse adagrad
#  we need to sync the step somehow
#  it seems like the mapping from local to optimizer lapse is still not correct


class DistAdagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        model,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        parameter_client=None,
        lapse_indexes=None,
        sync_levels=[],
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.lapse_optimizer_index_offset = (
            model.dataset.num_entities() + model.dataset.num_relations()
        )
        self.lapse_indexes = lapse_indexes
        # self.local_index_mappers = local_index_mappers
        self.local_index_mappers = [
            model._entity_embedder.local_index_mapper,
            model._relation_embedder.local_index_mapper,
        ]
        self.local_to_lapse_mappers = [
            model._entity_embedder.local_to_lapse_mapper,
            model._relation_embedder.local_to_lapse_mapper,
        ]
        self.pulled_parameters = [None, None]

        self.sync_levels = sync_levels

        self.parameter_client = parameter_client
        # these are numpy tensors in which we pull the current values from lapse to
        # update the optimizer
        # TODO: find a way that we don't have to store these parameters multiple times
        #  (lapse, optimizer, this tensor)
        self.lapse_update_tensors = [None, None]

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super(DistAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            if parameter_client.rank == 2 and parameter_client.get_lr == 0:
                self.parameter_client.set_lr(group["lr"])
            for i, p in enumerate(group["params"]):
                state = self.state[p]
                state["step"] = 0
                state["sum"] = torch.full_like(
                    p, initial_accumulator_value, memory_format=torch.preserve_format
                )
                # TODO: we need to find a good way to init complete lapse at the
                #  beginning, so that workers won't overwrite each other
                # initialize optimizer parameters in lapse
                # self.lapse_worker.push(self.optimizer_lapse_indexes[i],
                #                       np.full((len(self))))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                self.parameter_client.step_optim(i)
                state["step"] = self.parameter_client.get_step_optim(i)
                if self.parameter_client.rank == 2:
                    self.parameter_client.set_lr(group["lr"])
                group["lr"] = self.parameter_client.get_lr()

                # state["step"] += 1

                if group["weight_decay"] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(p, alpha=group["weight_decay"])

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if grad.is_sparse:
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_indices_flat = grad_indices.flatten()
                    grad_values = grad._values()
                    size = grad.size()

                    # pull the current internal optimizer parameters
                    if self.lapse_update_tensors[i] is None:
                        self.lapse_update_tensors[i] = np.zeros(size, dtype=np.float32)
                    # TODO: indexing on numpy update tensor creates a new tensor
                    #  updates will be written in the wrong tensor
                    #  sometimes the tensor is even freed before we even write in it
                    if self.sync_levels[i] == "batch":
                        update_mask = torch.zeros(
                            len(self.local_to_lapse_mappers[i]), dtype=torch.bool
                        )
                        update_mask[grad_indices_flat.cpu()] = True
                        update_tensor = torch.zeros(
                            (torch.sum(update_mask).item(), grad.size()[1]),
                            dtype=torch.float32,
                        )
                        keys_optim = (
                            self.local_to_lapse_mappers[i]
                            + self.lapse_optimizer_index_offset
                        )[update_mask]
                        self.parameter_client.pull(
                            keys_optim, update_tensor,
                        )
                        state["sum"][update_mask] = update_tensor.to(state["sum"].device)

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    sum_update_values = grad_values.pow(2)
                    state["sum"].add_(make_sparse(sum_update_values))
                    if self.sync_levels[i] == "batch":
                        self.parameter_client.push(
                            keys_optim, sum_update_values.cpu(),
                        )

                    std = state["sum"].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group["eps"])
                    update_value = (grad_values / std_values).mul_(-clr)
                    if self.sync_levels[i] == "batch":
                        self.parameter_client.push(
                            self.local_to_lapse_mappers[i][update_mask],
                            update_value.cpu(),
                        )
                    else:
                        p.add_(make_sparse(update_value))
                    # p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    # pull the current internal optimizer parameters
                    update_mask = self.local_to_lapse_mappers[i] != -1
                    update_tensor = torch.zeros(
                        (torch.sum(update_mask).item(), p.shape[1]), dtype=torch.float32
                    )
                    keys_optim = (
                        self.local_to_lapse_mappers[i]
                        + self.lapse_optimizer_index_offset
                    )[update_mask]
                    self.parameter_client.pull(keys_optim, update_tensor)
                    state["sum"][update_mask] = update_tensor.to(state["sum"].device)

                    # push the updated internal optimizer parameters to lapse
                    # state['sum'].addcmul_(grad, grad, value=1)
                    sum_update = grad * grad
                    self.parameter_client.push(
                        keys_optim, sum_update.cpu()[update_mask], asynchronous=True,
                    )
                    state["sum"].add_(sum_update)
                    std = state["sum"].sqrt().add_(group["eps"])

                    # we do not update the model parameters anymore, but push the
                    # updates to lapse
                    # p.addcdiv_(grad, std, value=-clr)
                    update_value = -clr * grad / std
                    self.parameter_client.push(
                        self.local_to_lapse_mappers[i][update_mask].astype(np.uint64),
                        update_value.cpu()[update_mask],
                        asynchronous=True,
                    )

        return loss

    def pull_entities(self, entity_ids):
        self._pull_parameters(entity_ids, 0)

    def pull_relations(self, relation_ids):
        self._pull_parameters(relation_ids, 1)

    def _pull_parameters(self, ids, parameter_idx):
        self.pulled_parameters[parameter_idx] = ids
        local_ids = self.local_index_mappers[parameter_idx][ids]
        lapse_ids = self.local_to_lapse_mappers[parameter_idx][local_ids]
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i != parameter_idx:
                    continue
                update_tensor = torch.zeros(
                    (len(ids), p.size()[1]), dtype=torch.float32
                )
                keys_optim = lapse_ids + self.lapse_optimizer_index_offset
                self.parameter_client.pull(
                    keys_optim, update_tensor,
                )
                self.state[p]["sum"][local_ids] = update_tensor.to(
                    self.state[p]["sum"].device
                )

    def set_entities(self):
        self._set_parameters(0)

    def set_relations(self):
        self._set_parameters(1)

    def _set_parameters(self, parameter_idx):
        pulled_parameter_ids = self.pulled_parameters[parameter_idx]
        lapse_optim_ids = (
            self.lapse_indexes[parameter_idx][pulled_parameter_ids]
            + self.lapse_optimizer_index_offset
        )
        local_ids = self.local_index_mappers[parameter_idx][pulled_parameter_ids]
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if i != parameter_idx:
                    continue
                self.parameter_client.set(
                    lapse_optim_ids, self.state[p]["sum"][local_ids]
                )

    def pull_all(self):
        # get all optimizer parameters out of lapse
        # only works if the optimizer has the complete size
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                update_tensor = torch.zeros_like(p)
                keys_optim = (
                    torch.arange(p.shape[0]) + self.lapse_optimizer_index_offset[i]
                )
                self.parameter_client.pull(keys_optim, update_tensor)
                state["sum"][:, :] = update_tensor

    def push_all(self):
        # push all optimizer parameters into lapse
        # only works if the optimizer has the complete size
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                keys_optim = (
                    torch.arange(p.shape[0]) + self.lapse_optimizer_index_offset[i]
                )
                self.parameter_client.push(keys_optim, state["sum"].cpu())
