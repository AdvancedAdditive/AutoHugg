from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    repo_name: str
    card_location: str


@dataclass(frozen=True)
class Config:
    title: str
    server: "Server"
    client: "Client"
    hub: "Hub"


@dataclass(frozen=True)
class Server:
    host: str
    port: int
    log_level: str


@dataclass(frozen=True)
class Model:
    repo_name: str
    card_location: str
    min_reward: int


@dataclass(frozen=True)
class Hub:
    endpoint: str
    username: str
    branch: str
    model: "Model"
    dataset: "Dataset"


@dataclass(frozen=True)
class Client:
    log_level: str
