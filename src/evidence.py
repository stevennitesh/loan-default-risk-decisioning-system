"""Content fingerprints for local scientific evidence; no release operations."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

from src.runtime import REPO_ROOT, resolve_config_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(config):
    paths = [
        path
        for folder, glob in (
            ("src", "*.py"),
            ("sql", "*.sql"),
            ("configs", "*.yaml"),
            ("tests", "*.py"),
        )
        for path in (REPO_ROOT / folder).glob(glob)
    ]
    paths += [
        REPO_ROOT / name
        for name in (
            "requirements.lock",
            "requirements.txt",
            "pyproject.toml",
            "Makefile",
        )
    ]
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "dirty_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        ),
        "source_sha256": {
            path.relative_to(REPO_ROOT).as_posix(): file_sha256(path)
            for path in sorted(paths)
        },
        "raw_sha256": {
            name: file_sha256(path) if path.exists() else None
            for name, filename in config["source_files"].items()
            for path in [resolve_config_path(config, "raw_dir") / filename]
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
    }
