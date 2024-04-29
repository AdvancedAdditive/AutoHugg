import logging
import os
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import stable_baselines3
import torch
from huggingface_hub import HfApi

from hugg.utils.config_types import Config
from hugg.utils.helpers import (get_hugg_token,
                                package_autoencoder_model_to_hub,
                                package_dataset_to_hub,
                                package_sb_model_to_hub)
from hugg.utils.load_config import load
from torch import nn
from typing import Any, Dict, Optional, Union
import requests


class HuggingFaceRepo:
    def __init__(self, config: Config, repo_type: str, branch_name: str,repo_name=None) -> None:
        """
        Initialize the HuggingFaceRepo object.

        Args:
            config (Config): The configuration object.
            repo_type (str): The type of repository ("model" or "dataset").
            branch_name (str): The name of the branch.
        """
        self.config = config
        self.token = get_hugg_token()
        self.branch_name = branch_name
        self.hub_reader = HfApi(token=self.token.read)
        self.hub_writer = HfApi(token=self.token.write)
        self.user_name = self.config.hub.username
        match repo_type:
            case "model":
                self.repo_type = "model"
                self.repo_config = self.config.hub.model
            case "dataset":
                self.repo_type = "dataset"
                self.repo_config = self.config.hub.dataset
            case None:
                self.repo_type = "model"
                self.repo_config = self.config.hub.model
            case _:
                raise ValueError("Repo type must be either 'model' or 'dataset'")
        if repo_name == None:
            self.repo_name = self.repo_config.repo_name
        else:
            self.repo_name = repo_name

        def check_internet_connection() -> bool:
            try:
                response = requests.get("https://www.google.com")
                return response.status_code == 200
            except requests.ConnectionError:
                return False

        if not check_internet_connection():
            raise ConnectionError("No internet connection available.")
        self.repo_id = self._create_model()
        self.checkout(self.branch_name)
 
    def _create_model(self) -> str:
        """
        Create a new model repository.

        Returns:
            str: The repository ID.
        """
        try:
            repo_id = f"{self.user_name}/{self.repo_name}"
            url = self.hub_writer.create_repo(repo_id, private=True, repo_type=self.repo_type, exist_ok=True)
            logging.info(f"Created repo at {url}")
        except Exception as e:
            logging.error(f"Error creating repo: {e}")
            return None
        return repo_id
    
    def checkout(self, branch: str) -> None:
        """
        Checkout the specified branch.

        Args:
            branch (str): The name of the branch.
        """
        try:
            self.hub_writer.create_branch(repo_id=self.repo_id, branch=branch, token=self.token.write, exist_ok=True, repo_type=self.repo_type)
        except Exception as e:
            logging.error(f"Error checking out branch: {e}")
        
    def push(self, tmp_dir: str, commit_message: str) -> None:
        """
        Push the model to the repository.

        Args:
            tmp_dir (str): The temporary directory containing the model files.
            commit_message (str): The commit message.
        """
        if not os.path.isdir(tmp_dir):
            raise ValueError(f"{tmp_dir} is not a valid directory")
        try:
            self.hub_writer.upload_folder(repo_id=self.repo_id,
                folder_path=tmp_dir,
                path_in_repo="",
                commit_message=commit_message,
                token=self.token.write,
                revision=self.branch_name,
                repo_type=self.repo_type)
            logging.info(f"Pushed model to {self.repo_id}")
        except Exception as e:
            logging.error(f"Error pushing model: {e}")
    
    def pull(self, filename: str, destination_dir: Optional[str] = None) -> Path:
        """
        Pull the model from the repository.

        Args:
            filename (str): The name of the file to pull.
            destination_dir (str, optional): The destination directory to save the pulled model. Defaults to None.

        Returns:
            Path: The path to the pulled model.
        """
        try:
            local_dir = self.hub_reader.hf_hub_download(repo_id=self.repo_id, filename=filename, revision=self.branch_name, token=self.token.read, local_dir=destination_dir, repo_type=self.repo_type,local_dir_use_symlinks=False)
        except Exception as e:
            logging.error(f"Error pulling model: {e}")
            return None
        logging.info(f"Pulled model from {self.repo_id} to {local_dir}")
        return local_dir

    def get_model_card_info(self, key: str) -> Optional[float]:
        """
        Get the model card information.

        Args:
            key (str): The key of the model card information to retrieve.

        Returns:
            Optional[float]: The model card information value, or None if not found.
        """
        try:
            if self.repo_type == "model":
                card = self.hub_reader.model_info(repo_id=self.repo_id, revision=self.branch_name, token=self.token.read)
            if self.repo_type == "dataset":
                card = self.hub_reader.dataset_info(repo_id=self.repo_id, revision=self.branch_name, token=self.token.read)
            last_commit = self.hub_reader.list_repo_commits(repo_id=self.repo_id, revision=self.branch_name, token=self.token.read,repo_type=self.repo_type)[0]

        except Exception as e:
            logging.error(f"Error getting model card info: {e}")
            return None
        if card.card_data is None:
            logging.warning(f"Model card not found for {self.repo_id} creating a new one?")
            return None
        match key:
            case "score":
                return float(card.card_data.eval_results[0].metric_value)
            case "last_commit":
                return last_commit.commit_id
            case "length":
                return int(card.card_data["model-index"][0]["results"][0]["metrics"][0]["value"])
            case _:
                raise NotImplementedError(f"Key {key} not implemented")
    


class HuggingFaceController:
    def __init__(self, config: Config, branch_name: str, repo_type: str,repo_name=None) -> None:
        """
        Initialize the HuggingFaceController object.

        Args:
            config (Config): The configuration object.
            branch_name (str): The name of the branch.
            repo_type (str): The type of repository ("model" or "dataset").
        """
        self.config = config
        self.branch_name = branch_name
        self.repo = HuggingFaceRepo(config, repo_type, branch_name,repo_name)
        
    def upload(self, commit_message: str, package_func: callable, *args, **kwargs) -> None:
        """
        Upload the model to the repository.

        Args:
            commit_message (str): The commit message.
            package_func (callable): The function to package the model.
            *args: Additional arguments for the packaging function.
            **kwargs: Additional keyword arguments for the packaging function.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            package_func(*args, tmpdir, **kwargs)
            self.repo.push(tmpdir, commit_message=commit_message)

    def get_model_card_info(self, key: str) -> Optional[float]:
        """
        Get the model card information.

        Args:
            key (str): The key of the model card information to retrieve.

        Returns:
            Optional[float]: The model card information value, or None if not found.
        """
        return self.repo.get_model_card_info(key)
    
    def download_latest(self, name: str, destination_dir: Optional[str] = None) -> Path:
        """
        Download the latest model from the repository.

        Args:
            name (str): The name of the file to download.
            destination_dir (str, optional): The destination directory to save the downloaded model. Defaults to None.

        Returns:
            Path: The path to the downloaded model.
        """
        _dir = self.repo.pull(name, destination_dir)
        commit_id = self.repo.get_model_card_info("last_commit")
        if destination_dir is not None and commit_id is not None:
            commit_file = os.path.join(destination_dir, commit_id)
            with open(commit_file, "w") as f:
                pass
        return _dir


class HuggingFaceStableBaseLinesModelController:
    def __init__(self, config: Config, branch_name: str) -> None:
        """
        Initialize the HuggingFaceStableBaseLinesModelController object.

        Args:
            config (Config): The configuration object.
            branch_name (str): The name of the branch.
        """
        self.controller = HuggingFaceController(config, branch_name, "model")

    def get_current_score(self) -> Optional[float]:
        """
        Get the current model score.

        Returns:
            Optional[float]: The current model score, or None if not found.
        """
        score = self.controller.get_model_card_info("score")
        return score if score is not None else -1 * np.inf

    def upload_model(self, sb_model: stable_baselines3.PPO, env_id: str, train_config: dict, score: float, force_update: bool = False) -> None:
        """
        Upload the stable baselines model to the repository.

        Args:
            sb_model (stable_baselines3.PPO): The stable baselines model.
            env_id (str): The environment ID.
            train_config (dict): The training configuration.
            score (float): The model score.
            force_update (bool, optional): Whether to force update the model. Defaults to False.
        """
        model_name = "sb_model"
        if force_update or score > self.get_current_score():
            commit_message = f"AutoPush with SB Model {model_name} trained on {env_id} with score {score}"
            self.controller.upload(commit_message, package_sb_model_to_hub, sb_model, model_name, train_config, env_id, score)
        else:
            logging.info(f"Model {model_name} did not meet the minimum score requirement")

        

    def download_latest_model(self) -> Optional[stable_baselines3.PPO]:
        """
        Download the latest model from the repository.

        Returns:
            Optional[stable_baselines3.PPO]: The downloaded stable baselines model, or None if not found.
        """
        local_dir = self.controller.download_latest("sb_model.zip")
        with open(local_dir, "rb") as f:
            try:
                return stable_baselines3.PPO.load(f)
            except:
                logging.warning("Could not load model as stable baselines model")
            raise ValueError("Could not load model. Check the model type should be stable baselines model")

class HuggingFaceAutoencoderModelController:
    def __init__(self, config: Config, branch_name: str) -> None:
        """
        Initialize the HuggingFaceAutoencoderModelController object.

        Args:
            config (Config): The configuration object.
            branch_name (str): The name of the branch.
        """
        self.controller = HuggingFaceController(config, branch_name, "model")

    def get_current_score(self) -> Optional[float]:
        """
        Get the current model score.

        Returns:
            Optional[float]: The current model score, or None if not found.
        """
        score = self.controller.get_model_card_info("score")
        return score if score is not None else 1 * np.inf

    def upload_model(self, torch_model: nn.Module, dataset_id: str, train_config: dict, score: float, force_update: bool = False) -> None:
        """
        Upload the torch model to the repository.

        Args:
            torch_model (nn.Module): The torch model.
            dataset_id (str): The dataset ID.
            train_config (dict): The training configuration.
            score (float): The model score.
            force_update (bool, optional): Whether to force update the model. Defaults to False.
        """
        model_name = "autoencoder.model"
        if force_update or score < self.get_current_score():
            commit_message = f"AutoPush with Autoencoder Model {model_name} with score {score}"
            self.controller.upload(commit_message, package_autoencoder_model_to_hub, torch_model, model_name, train_config, dataset_id, score)
        else:
            logging.info(f"Model {model_name} did not meet the minimum score requirement")

    def download_latest_model(self) -> Optional[nn.Module]:
        """
        Download the latest model from the repository.

        Returns:
            Optional[nn.Module]: The downloaded torch model, or None if not found.
        """
        local_dir = self.controller.download_latest("autoencoder.model")
        with open(local_dir, "rb") as f:
            try:
                return torch.load(f)
            except:
                raise ValueError("Could not load model. Check the model type should be stable baselines model")      

class HuggingFaceDatasetController:
    def __init__(self, config: Config, branch_name: str,repo_name=None) -> None:
        """
        Initialize the HuggingFaceDatasetController object.

        Args:
            config (Config): The configuration object.
            branch_name (str): The name of the branch.
        """
        self.controller = HuggingFaceController(config, branch_name, "dataset",repo_name)
        self.dataset_id: str = self.controller.branch_name + ".db"

    def upload_dataset(self, dataset_location: Path, dataset_config: Any, length: int) -> None:
        """
        Upload the dataset to the repository.

        Args:
            dataset_location (Path): The location of the dataset.
            dataset_config (Any): The dataset configuration.
            length (int): The length of the dataset.
        """
        commit_message = f"AutoPush for Dataset {self.dataset_id} with length {length}"
        self.controller.upload(commit_message, package_dataset_to_hub, dataset_location, self.dataset_id, dataset_config, length)

    def download_latest_dataset(self, destination_dir: Optional[str] = None) -> Path:
        """
        Download the latest dataset from the repository.

        Args:
            destination_dir (str, optional): The destination directory to save the downloaded dataset. Defaults to None.

        Returns:
            Path: The path to the downloaded dataset.
        """
        if destination_dir is None:
            destination_dir = tempfile.mkdtemp()
        commit_id = self.controller.get_model_card_info("last_commit")

        if not os.path.exists(destination_dir):
            local_dir = self.controller.download_latest(self.dataset_id + ".zip", destination_dir)
        else:
            if not commit_id in os.listdir(destination_dir):
                local_dir = self.controller.download_latest(self.dataset_id + ".zip", destination_dir)
            else:
                local_dir = os.path.join(destination_dir, self.dataset_id )
                logging.info("Skiping Download. Latest Dataset already exists")
                return local_dir

        logging.info('decompressing the dataset')
        with zipfile.ZipFile(local_dir, 'r') as zip_ref:
            zip_ref.extractall(Path(local_dir).parent / self.dataset_id)
        if os.path.isfile(local_dir):
            os.remove(local_dir)
            
        return local_dir[:-4]
    
    def info(self):
        return self.controller.get_model_card_info("last_commit"),self.controller.get_model_card_info("length")


if __name__ == "__main__":
    pass
    # from dataclasses import dataclass

    # import torch
    # import torch.nn as nn


    # class DummyModel(nn.Module):
    #     def __init__(self):
    #         super(DummyModel, self).__init__()
    #         self.fc = nn.Linear(10, 1)

    #     def forward(self, x):
    #         return self.fc(x)
    # @dataclass   
    # class DummyConfig:
    #     test_val:int


    # # Instantiate the model
    # torch_model = DummyModel()
    # sb_model = stable_baselines3.PPO("MlpPolicy", "CartPole-v1")



    config = load()
    logging.basicConfig(level=logging.INFO)
    # model_repo = HuggingFaceDatasetController(config,"test_dataset")
    # model_repo.upload_dataset("./test/dataset/testdataset.db", DummyConfig(1), 1)
    # print(model_repo.download_latest_dataset("./"))

    c = HuggingFaceDatasetController(config,"GridNoiseEnv_36",repo_name ='autoencoder_data')
    c.download_latest_dataset('./data/datasets/hugg/tmp')
