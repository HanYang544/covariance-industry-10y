"""Kenneth French industry portfolio data interfaces."""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MISSING_MARKERS = (-99.99, -999.0)


def load_daily_returns(path: str | Path) -> pd.DataFrame:
    """Load a normalized CSV with a date column and decimal return columns."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True, na_values=MISSING_MARKERS)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Return data contains duplicate dates.")
    return frame.astype(float)


def validate_return_matrix(returns: pd.DataFrame) -> None:
    """Raise when a return matrix is empty or contains unusable values."""
    if returns.empty:
        raise ValueError("Return matrix is empty.")
    if returns.isna().any().any():
        raise ValueError("Return matrix contains missing values.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Return matrix contains infinite values.")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("Return dates must be unique and increasing.")


SOURCE_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "10_Industry_Portfolios_daily_CSV.zip"
)


def acquire_industry_10(raw_dir: Path) -> tuple[Path, dict]:
    """Reuse an immutable download or acquire the official ZIP and record provenance."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "10_Industry_Portfolios_daily_CSV.zip"
    if not archive.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            content = response.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zipped:
            zipped.testzip()
        archive.write_bytes(content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = raw_dir / "industry_10_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != digest:
            raise ValueError("Raw data checksum differs from its frozen manifest.")
    else:
        manifest = {
            "url": SOURCE_URL,
            "sha256": digest,
            "acquired_at_utc": datetime.fromtimestamp(
                archive.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "portfolio_weighting": "value_weighted",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return archive, manifest


def parse_industry_zip(archive: Path) -> tuple[pd.DataFrame, str]:
    """Read only the official value-weighted daily section; convert percent to decimal."""
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected one CSV member in industry archive.")
        lines = zipped.read(names[0]).decode("utf-8-sig").splitlines()
    section = next(
        i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Daily" in line
    )
    header_index = next(i for i in range(section + 1, len(lines)) if lines[i].startswith(","))
    columns = [item.strip() for item in next(csv.reader([lines[header_index]]))[1:]]
    dates, values = [], []
    for line in lines[header_index + 1:]:
        fields = next(csv.reader([line]))
        if not fields or not fields[0].strip().isdigit() or len(fields[0].strip()) != 8:
            break
        row = [float(item) for item in fields[1:]]
        if len(row) != len(columns):
            raise ValueError("Inconsistent asset count in raw data.")
        dates.append(fields[0].strip())
        values.append(row)
    frame = pd.DataFrame(values, index=pd.to_datetime(dates, format="%Y%m%d"), columns=columns)
    frame = frame.replace(list(MISSING_MARKERS), np.nan) / 100.0
    frame.index.name = "date"
    return frame, lines[0]
