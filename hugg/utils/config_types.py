from dataclasses import dataclass
from datetime import datetime
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Model:
    repo_name: str
    card_location: str
    min_reward: int


@dataclass(frozen=True)
class Dataset:
    repo_name: str
    card_location: str


@dataclass(frozen=True)
class Hub:
    endpoint: str
    username: str
    branch: str
    model: "Model"
    dataset: "Dataset"


@dataclass(frozen=True)
class Config:
    title: str
    hub: "Hub"
