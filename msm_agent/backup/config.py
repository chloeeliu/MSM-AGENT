from dataclasses import dataclass
from pathlib import Path
import yaml

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)