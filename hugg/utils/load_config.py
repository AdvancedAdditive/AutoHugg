import os
from pathlib import Path
from typing import Optional

from heracless import load_config

CONFIG_PATH = Path(os.path.dirname(__file__)).parent.parent / Path("./config/")
DUMP_PATH = Path(os.path.dirname(__file__)).parent.parent / Path("./hugg/utils/config_types.py")
if "PROOF_CONFIG" in os.environ:
    env_var = os.environ["PROOF_CONFIG"]
else:
    env_var = "dev"


def load() -> "Config":
    return load_config(
        cfg_path=CONFIG_PATH / Path(f"{env_var}.config.yaml"), dump_dir=DUMP_PATH, make_dir=True, frozen=True
    )


if __name__ == "__main__":
    config = load()
    print(config)
