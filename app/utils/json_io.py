import json
import time
from pathlib import Path


def read_json(path: str | Path):
    file_path = Path(path)
    for attempt in range(3):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if attempt >= 2:
                raise
            time.sleep(0.05 * (attempt + 1))


def write_json(path: str | Path, data) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    for attempt in range(3):
        try:
            tmp_path.replace(file_path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
