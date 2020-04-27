import os
import torch
from kge import Config, Dataset
from kge.util import load_checkpoint


def add_package_parser(subparsers):
    """Creates the parser for the command package"""
    package_parser = subparsers.add_parser(
        "package", help="Create packaged model (checkpoint only containing model)",
    )
    package_parser.add_argument("checkpoint", type=str, help="filename of a checkpoint")
    package_parser.add_argument(
        "--file", type=str, help="output filename of packaged model"
    )


def package_model(checkpoint_path: str, filename: str = None):
    """
    Converts a checkpoint to a packaged model.
    A packaged model only contains the model, entity/relation ids and the config.
    """
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    if checkpoint["type"] != "train":
        raise ValueError("Can only package trained checkpoints.")
    config = Config.create_from(checkpoint)
    dataset = Dataset.create_from(checkpoint, config, preload_data=False)
    packaged_model = {
        "type": "package",
        "model": checkpoint["model"],
    }
    packaged_model = config.save_to(packaged_model)
    packaged_model = dataset.save_to(packaged_model, ["entity_ids", "relation_ids"],)
    if filename is None:
        output_folder, filename = os.path.split(checkpoint_path)
        if "checkpoint" in filename:
            filename = filename.replace("checkpoint", "model")
        else:
            filename = filename.split(".pt")[0] + "_package.pt"
        filename = os.path.join(output_folder, filename)
    torch.save(packaged_model, filename)
