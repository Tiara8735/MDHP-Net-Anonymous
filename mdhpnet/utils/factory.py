import importlib
from easydict import EasyDict as edict
import os
import os.path as osp

import torch
from torch import nn


# [TODO][Enhancement] @tiara8735
# |- Maybe to use registry pattern to make the code more elegant?
class Registry:
    """
    A common registry for registering classes or functions.
    """

    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        """
        Register a class or function.
        """

        def wrapper(cls):
            self._registry[name] = cls
            return cls

        return wrapper

    def get(self, name: str):
        """
        Get the type of a class or function.
        """
        return self._registry[name]


def buildAnything(cfg: dict) -> object:
    """
    Build an object from the configuration.

    Parameters
    ----------
    cfg : dict
        Configuration.
        Must contain the following keys:
            typename: str
            module: str
            params: dict

    Returns
    -------
    object
        Object.
    """
    cfg = edict(cfg)
    module = importlib.import_module(cfg.module)
    typename = cfg.typename
    class_ = getattr(module, typename)
    params = cfg.params
    return class_(**params)


def buildModel(model_cfg: dict) -> nn.Module:
    """
    Build a model from the configuration.

    Parameters
    ----------
    model_cfg : dict
        Model configuration.
        Must contain the following keys:
            module: str
            typename: str
            params: dict
            ckpt_path: str | None

    Returns
    -------
    Module
        Model.
    """
    model_cfg = edict(model_cfg)
    model_typename = model_cfg.typename
    model_module = importlib.import_module(f"mdhpnet.models")
    model_class = getattr(model_module, model_typename)
    model_params = model_cfg.params
    model: torch.nn.Module = model_class(**model_params)

    # Return a model without loading the checkpoint if "ckpt_path" is not in the
    # configuration:
    if "ckpt_path" not in model_cfg:
        return model

    ckpt_path = model_cfg.ckpt_path
    if ckpt_path is None:
        n_seq = model.n_seq
        model_name = model.__class__.__name__
        # We assume that the checkpoint is saved in the following directory:
        # "./results/train/{model_name}_N{n_seq}/ckpt"
        supposed_ckpt_dir = osp.join(
            ".", "results", "train", f"{model_name}_N{n_seq}", "ckpt"
        )
        # Find a pth file starting with "val_best" in the directory
        for file in os.listdir(supposed_ckpt_dir):
            if file.startswith("val_best"):
                ckpt_path = osp.join(supposed_ckpt_dir, file)
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint file found in {supposed_ckpt_dir}.")

    # [TODO][Enhancement] @tiara8735
    # |- Add a logger here to log the loading of the checkpoint.

    # Load the checkpoint
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model


def buildCriterion(criterion_cfg: dict) -> nn.Module:
    """
    Build a criterion from the configuration.

    Parameters
    ----------
    criterion_cfg : dict
        Criterion configuration.
        Must contain the following keys:
            typename: str
            params: dict

    Returns
    -------
    Module
        Criterion.
    """
    criterion_cfg = edict(criterion_cfg)
    criterion_typename = criterion_cfg.typename
    criterion_module = importlib.import_module(f"mdhpnet.losses")
    criterion_class = getattr(criterion_module, criterion_typename)
    criterion_params = criterion_cfg.params
    return criterion_class(**criterion_params)


def buildOptimizer(optimizer_cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from the configuration.

    Parameters
    ----------
    optimizer_cfg : dict
        Optimizer configuration.
        Must contain the following keys:
            typename: str
            params: dict
    model : Module
        Model.

    Returns
    -------
    Optimizer
        Optimizer.
    """
    optimizer_cfg = edict(optimizer_cfg)
    optimizer_typename = optimizer_cfg.typename
    optimizer_module = importlib.import_module("torch.optim")
    optimizer_class = getattr(optimizer_module, optimizer_typename)
    optimizer_params = optimizer_cfg.params
    return optimizer_class(model.parameters(), **optimizer_params)


def buildDataset(dataset_cfg: dict) -> torch.utils.data.Dataset:
    """
    Build a dataset from the configuration.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration.
        Must contain the following keys:
            typename: str
            params: dict

    Returns
    -------
    Dataset
        Dataset.
    """
    dataset_cfg = edict(dataset_cfg)
    dataset_typename = dataset_cfg.typename
    dataset_module = importlib.import_module(f"mdhpnet.datasets")
    dataset_class = getattr(dataset_module, dataset_typename)
    dataset_params = dataset_cfg.params
    return dataset_class(**dataset_params)
