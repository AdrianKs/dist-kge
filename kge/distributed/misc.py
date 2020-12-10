from kge import Config

MIN_RANK = 2


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
