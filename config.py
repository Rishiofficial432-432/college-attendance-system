# config.py ---------------------------------------------------------
import yaml
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    if not _CFG_PATH.is_file():
        return {}
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


cfg = load_config()
