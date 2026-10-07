"""Settings from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """A setting has a value that the package cannot use."""


def load_dotenv(path: str | os.PathLike = ".env") -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        value = value.strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            added += 1
    return added


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    seed: int
    device: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        raw_seed = (env.get("DERMAVIT_SEED") or "42").strip()
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ConfigError(f"DERMAVIT_SEED must be an integer, got {raw_seed!r}") from exc
        device = (env.get("DERMAVIT_DEVICE") or "auto").strip().lower()
        if device not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError(f"DERMAVIT_DEVICE must be auto, cpu, cuda or mps, got {device!r}")
        return cls(
            data_dir=Path((env.get("DERMAVIT_DATA_DIR") or "data").strip()),
            output_dir=Path((env.get("DERMAVIT_OUTPUT_DIR") or "runs").strip()),
            seed=seed,
            device=device,
        )
