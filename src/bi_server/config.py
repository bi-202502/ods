import os
from pathlib import Path


def _get_root_path() -> Path:
    root = os.environ.get("BI_ROOT", None)
    return Path(root) if root else Path.cwd() / "data"


ROOT_PATH = _get_root_path()
