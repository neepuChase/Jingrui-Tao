"""Data loading utilities for power load analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def find_csv_files(repo_root: Path) -> List[Path]:
    """Find candidate CSV files in repository, excluding generated folders."""
    excluded_dirs = {".git", "outputs", "figures", "__pycache__", ".venv", "venv"}
    csv_files: List[Path] = []

    for path in repo_root.rglob("*.csv"):
        if any(part in excluded_dirs for part in path.parts):
            continue
        csv_files.append(path)

    return sorted(csv_files)


def detect_encoding(file_path: Path) -> str:
    """Try common encodings and return the first one that works."""
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "latin1"]

    for enc in encodings:
        try:
            pd.read_csv(file_path, encoding=enc, nrows=20)
            return enc
        except Exception:
            continue

    return "utf-8"


def load_csv_robust(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load CSV with robust encoding handling."""
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    return df, encoding


def choose_primary_dataset(repo_root: Path) -> Path:
    """Choose the most likely primary dataset CSV file."""
    csv_files = find_csv_files(repo_root)
    if not csv_files:
        raise FileNotFoundError("No CSV files were found in this repository.")

    # Prefer files inside data/ and with larger sizes.
    csv_files = sorted(
        csv_files,
        key=lambda p: (
            0 if "data" in p.parts else 1,
            -p.stat().st_size,
            p.name,
        ),
    )
    return csv_files[0]
