import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def settings():
    return read_json("config.json")


def db_path(index):
    if index not in {"baseline", "rebuilt"}:
        raise ValueError("Index must be baseline or rebuilt")
    # Only the working copy is opened by Chroma; the archived snapshot stays untouched.
    return ROOT / "data" / "chroma" / ("working" if index == "baseline" else "rebuilt")


def collection_name(index):
    return settings()["baseline_collection" if index == "baseline" else "rebuilt_collection"]
