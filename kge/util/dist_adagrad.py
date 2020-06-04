import torch
import numpy as np
from torch.optim.optimizer import Optimizer

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

    def __init__(self, model, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, lapse_worker=None, lapse_indexes=None):
        params = [p for p in model.parameters() if p.requires_grad]
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.lapse_optimizer_index_offset = model.dataset.num_entities() + model.dataset.num_relations()
        self.lapse_indexes = lapse_indexes
        # self.local_index_mappers = local_index_mappers
        self.local_index_mappers = [model._entity_embedder.local_index_mapper,
                                    model._relation_embedder.local_index_mapper]
        self.local_to_lapse_mappers = [model._entity_embedder.local_to_lapse_mapper, model._relation_embedder.local_to_lapse_mapper]

        self.lapse_worker = lapse_worker
        self.optimizer_lapse_indexes = [np.arange(model.dataset.num_entities() + model.dataset.num_relations(), 2*model.dataset.num_entities() + model.dataset.num_relations()),
                                        np.arange(2*model.dataset.num_entities() + model.dataset.num_relations(), 2*model.dataset.num_entities() + 2*model.dataset.num_relations())]
        # these are numpy tensors in which we pull the current values from lapse to
        # update the optimizer
        # TODO: find a way that we don't have to store these parameters multiple times
        #  (lapse, optimizer, this tensor)
        self.lapse_update_tensors = [None, None]

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(DistAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, initial_accumulator_value, memory_format=torch.preserve_format)
                # intialize optimizer parameters in lapse
                self.lapse_worker.push(self.optimizer_lapse_indexes[i],
                                       state['sum'].cpu().numpy())

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

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
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(p, alpha=group['weight_decay'])

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum'].sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(group['eps'])
                    p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    # pull the current internal optimizer parameters
                    if self.lapse_update_tensors[i] is None:
                        self.lapse_update_tensors[i] = np.zeros(p.shape, dtype=np.float32)
                    self.lapse_worker.pull(self.local_to_lapse_mappers[i].astype(np.long) + self.lapse_optimizer_index_offset, self.lapse_update_tensors[i])
                    state['sum'][:, :] = torch.from_numpy(self.lapse_update_tensors[i]).to(state['sum'].device)

                    # push the updated internal optimizer parameters to lapse
                    # state['sum'].addcmul_(grad, grad, value=1)
                    sum_update = grad*grad
                    self.lapse_worker.push(self.local_to_lapse_mappers[i] + self.lapse_optimizer_index_offset, sum_update.cpu().numpy())
                    state['sum'] += sum_update
                    std = state['sum'].sqrt().add_(group['eps'])

                    # we do not update the model parameters anymore, but push the
                    # updates to lapse
                    # p.addcdiv_(grad, std, value=-clr)
                    update_value = -clr * grad/std
                    self.lapse_worker.push(self.local_to_lapse_mappers[i], update_value.cpu().numpy())

        return loss