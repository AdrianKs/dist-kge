from kge import Config


def get_min_rank(config: Config):
    if config.get("job.distributed.parameter_server") == "shared":
        # with a shared parameter server we don't create an additional process
        return 1
    else:
        return 2


def get_optimizer_dim(config: Config, dim):
    optimizer = config.get("train.optimizer.default.type")
    if optimizer == "dist_sgd":
        optimizer_dim = -1
    elif optimizer == "dist_adagrad":
        optimizer_dim = dim
    elif optimizer == "dist_rowadagrad":
        optimizer_dim = 1
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented in distributed setting")
    return optimizer_dim
