import json
from pathlib import Path


PROJECT_ROOT = Path.cwd()


def load_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(config):
    path = Path(config["manifest_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_data_files(manifest):
    for record in manifest["files"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"file size does not match manifest: {path}")
