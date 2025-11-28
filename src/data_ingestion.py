import os
import math
import datetime as dt
from typing import Optional

import pandas as pd
import requests

from .config import (
    CPD_CRIMES_ENDPOINT,
    SOCIOECONOMIC_CSV_URL,
    COMMUNITY_AREAS_GEOJSON_URL,
    RAW_DIR,
    RAW_CPD_FILE,
    RAW_SOCIO_FILE,
    RAW_COMMUNITY_AREAS_GEOJSON,
)


def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    os.makedirs(RAW_DIR, exist_ok=True)


def _socrata_fetch_json(
    endpoint: str,
    where: Optional[str] = None,
    select: Optional[str] = None,
    limit: int = 50000,
    max_pages: int = 50,
) -> pd.DataFrame:
    """
    Fetch paginated JSON from a Socrata endpoint and return a DataFrame.

    The function uses app-token-less access and is suitable for moderate pulls.
    """
    params = {}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    params["$limit"] = limit

    frames = []
    for page in range(max_pages):
        params["$offset"] = page * limit
        resp = requests.get(endpoint, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        frames.append(pd.DataFrame(data))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def fetch_cpd_crimes(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_rows: int = 250_000,
) -> pd.DataFrame:
    """
    Download Chicago crimes within a date range and save to raw parquet.

    Dates must be ISO timestamps (e.g., '2024-01-01T00:00:00.000'). If omitted,
    pulls approximately the last 365 days.
    """
    ensure_dirs()
    if not start_date or not end_date:
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=365)
        start_date = start.strftime("%Y-%m-%dT%H:%M:%S.000")
        end_date = end.strftime("%Y-%m-%dT%H:%M:%S.000")

    where = f"date >= '{start_date}' AND date <= '{end_date}'"
    select = \
        "id,date,primary_type,description,location_description,arrest,domestic," \
        "beat,district,ward,community_area,latitude,longitude"

    page_limit = 50000
    pages = math.ceil(max_rows / page_limit)
    df = _socrata_fetch_json(
        CPD_CRIMES_ENDPOINT, where=where, select=select, limit=page_limit, max_pages=pages
    )
    if df.empty:
        return df
    # Normalize types
    for c in ["arrest", "domestic"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
    # Save
    df.to_csv(RAW_CPD_FILE, index=False)
    return df


def fetch_socioeconomic() -> pd.DataFrame:
    """
    Download socioeconomic indicators by community area and save CSV.
    """
    ensure_dirs()
    resp = requests.get(SOCIOECONOMIC_CSV_URL, timeout=60)
    resp.raise_for_status()
    with open(RAW_SOCIO_FILE, "wb") as f:
        f.write(resp.content)
    df = pd.read_csv(RAW_SOCIO_FILE)
    return df


def fetch_community_areas_geojson() -> dict:
    """
    Download community areas boundaries GeoJSON and save to file.
    """
    ensure_dirs()
    resp = requests.get(COMMUNITY_AREAS_GEOJSON_URL, timeout=60)
    resp.raise_for_status()
    with open(RAW_COMMUNITY_AREAS_GEOJSON, "wb") as f:
        f.write(resp.content)
    return resp.json()
