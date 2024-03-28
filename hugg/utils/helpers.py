import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PosixPath
from typing import Generator, Union

import gymnasium as gym
import torch
import yaml
from huggingface_hub import ModelCard, ModelCardData
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from hugg.utils.modelcard_helper import (generate_autoencoder_model_card,
                                         generate_dataset_card,
                                         generate_sb_model_card, save_card)


@dataclass
class HuggingFaceApiToken:
    read: str
    write: str


def _get_read_token():
    """
    Retrieves the read token from the environment variable READ_HUGG_TOKEN.

    Raises:
        ValueError: If the environment variable READ_HUGG_TOKEN is not set.

    Returns:
        str: The read token.
    """
    token = os.getenv("READ_HUGG_TOKEN")
    if token is None:
        raise ValueError("Environment Variable READ_HUGG_TOKEN not set")
    return token


def _get_write_token():
    """
    Retrieves the write token from the environment variable WRITE_HUGG_TOKEN.

    Raises:
        ValueError: If the environment variable WRITE_HUGG_TOKEN is not set.

    Returns:
        str: The write token.
    """
    token = os.getenv("WRITE_HUGG_TOKEN")
    if token is None:
        raise ValueError("Environment Variable WRITE_HUGG_TOKEN not set")
    return token


def get_hugg_token():
    """
    Retrieves the Hugging Face API token for reading and writing.

    Returns:
        HuggingFaceApiToken: An instance of the HuggingFaceApiToken class containing the read and write tokens.
    """
    r_token = _get_read_token()
    w_token = _get_write_token()
    return HuggingFaceApiToken(read=r_token, write=w_token)


def _generate_config(config: "Config", tmpdirname: Path) -> None:
    """
    Generate a config file in YAML format.

    Args:
        config (Config): The configuration object.
        tmpdirname (Path): The directory where the config file will be saved.

    Raises:
        ValueError: If tmpdirname is not a string or a Path object.
    """
    match tmpdirname:
        case Path():
            pass
        case str():
            tmpdirname = Path(tmpdirname)
        case PosixPath():
            tmpdirname = Path(tmpdirname)
        case _:
            raise ValueError("tmpdirname must be a string or a Path object")
    with open(tmpdirname / "config.yml", 'w') as file:
        yaml.dump(asdict(config), file)


def save_model(model: Union[BaseAlgorithm, torch.nn.Module], model_name: str, tmpdir: Path) -> None:
    """
    Save the model to the specified directory.

    Args:
        model (Union[BaseAlgorithm, torch.nn.Module]): The model to be saved.
        model_name (str): The name of the model.
        tmpdir (Path): The directory where the model will be saved.

    Raises:
        ValueError: If the model is not a stable-baselines3 model or a torch.nn.Module.
    """
    match model:
        case BaseAlgorithm():
            model.save(tmpdir / model_name)
        case torch.nn.Module():
            torch.save(model.state_dict(), tmpdir / model_name)
        case _:
            raise ValueError("Model must be a stable-baselines3 model or a torch.nn.Module")

def package_sb_model_to_hub(model: BaseAlgorithm, model_name: str, config: "Config", env_id: str, score: float, tmpdir: Path) -> None:
    """
    Package and save a stable-baselines3 model to the Hugging Face Model Hub.

    Args:
        model (BaseAlgorithm): The stable-baselines3 model to be saved.
        model_name (str): The name of the model.
        config (Config): The configuration object.
        env_id (str): The ID of the environment.
        score (float): The score of the model.
        tmpdir (Path): The directory where the model will be saved.

    Raises:
        ValueError: If the model is not a stable-baselines3 model.
    """

    # Step 1: Save the model
    save_model(model, model_name, tmpdir)

    # Step 2: Create a config file
    _generate_config(config, tmpdir)

    # Step 5: Generate the model card
    generated_model_card, metadata = generate_sb_model_card(model_name, env_id, score)
    save_card(tmpdir, generated_model_card, metadata)
    logging.info(f"Model {model_name} packaged successfully and saved to {tmpdir}")

def package_autoencoder_model_to_hub(model: BaseAlgorithm, model_name: str, config: "Config", dataset_id: str, score: float, tmpdir: Path) -> None:
    """
    Package and save an autoencoder model to the Hugging Face Model Hub.

    Args:
        model (BaseAlgorithm): The autoencoder model to be saved.
        model_name (str): The name of the model.
        config (Config): The configuration object.
        dataset_id (str): The ID of the dataset.
        score (float): The score of the model.
        tmpdir (Path): The directory where the model will be saved.

    Raises:
        ValueError: If the model is not a stable-baselines3 model.
    """

    # Step 1: Save the model
    save_model(model, model_name, tmpdir)

    # Step 2: Create a config file
    _generate_config(config, tmpdir)

    # Step 5: Generate the model card
    generated_model_card, metadata = generate_autoencoder_model_card(model_name, dataset_id, score)
    save_card(tmpdir, generated_model_card, metadata)
    logging.info(f"Model {model_name} packaged successfully and saved to {tmpdir}")

def save_dataset(dataset_location: Path, dataset_id: str, tmpdir: Path) -> None:
    """
    Save the dataset to the specified directory.

    Args:
        dataset_location (Path): The location of the dataset.
        dataset_id (str): The ID of the dataset.
        tmpdir (Path): The directory where the dataset will be saved.

    Raises:
        ValueError: If the dataset location does not exist.
    """
    if not os.path.exists(dataset_location):
        raise ValueError(f"Dataset location {dataset_location} does not exist")
    shutil.make_archive(tmpdir / dataset_id, 'zip', dataset_location)

def package_dataset_to_hub(dataset_location: Path, dataset_id: str, config: "Config", length: int, tmpdir: Path) -> None:
    """
    Package and save a dataset to the Hugging Face Model Hub.

    Args:
        dataset_location (Path): The location of the dataset.
        dataset_id (str): The ID of the dataset.
        config (Config): The configuration object.
        length (int): The length of the dataset.
        tmpdir (Path): The directory where the dataset will be saved.

    Raises:
        ValueError: If the dataset location does not exist.
    """
    # Step 1: Save the dataset
    save_dataset(dataset_location, dataset_id, tmpdir)

    # Step 2: Create a config file
    _generate_config(config, tmpdir)

    # Step 5: Generate the dataset card
    generated_dataset_card, metadata = generate_dataset_card(dataset_id, length)
    save_card(tmpdir, generated_dataset_card, metadata)
    logging.info(f"Dataset {dataset_id} packaged successfully and saved to {tmpdir}")
