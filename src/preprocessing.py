import os
import json
from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    RAW_CPD_FILE,
    RAW_SOCIO_FILE,
    CLEANED_CPD_FILE,
    AGG_MONTHLY_FILE,
    GUN_KEYWORDS,
)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'date' column and derive time features."""
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()
    df["month_start"] = df["date"].dt.to_period("M").dt.start_time
    return df


def _gun_related_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic to flag gun-related incidents using description and primary type."""
    desc = df.get("description", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    prim = df.get("primary_type", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
    keyword_hit = desc.apply(lambda s: any(k in s for k in GUN_KEYWORDS))
    prim_hit = prim.str.contains("WEAPONS VIOLATION|HOMICIDE|BATTERY|ASSAULT|ROBBERY", regex=True)
    df["gun_related"] = keyword_hit | prim_hit
    return df


def load_and_clean() -> pd.DataFrame:
    """
    Load raw CPD crimes and clean, derive features, and save cleaned parquet.
    """
    if not os.path.exists(RAW_CPD_FILE):
        raise FileNotFoundError(f"Raw CPD file not found at {RAW_CPD_FILE}")
    df = pd.read_csv(RAW_CPD_FILE)
    df = _standardize_columns(df)
    df = _parse_dates(df)
    df = _gun_related_flag(df)
    # Drop obvious duplicates by id + date
    if "id" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset=["id"], keep="last")
    df.to_csv(CLEANED_CPD_FILE, index=False)
    return df


def build_monthly_by_community() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate monthly incident counts by community area and merge socioeconomic data.
    Returns (agg_df, socio_df).
    """
    crimes = pd.read_csv(CLEANED_CPD_FILE)
    crimes = crimes[crimes["gun_related"]]
    # Keep rows with community_area
    crimes = crimes.dropna(subset=["community_area"]).copy()
    crimes["community_area"] = pd.to_numeric(crimes["community_area"], errors="coerce")
    crimes = crimes.dropna(subset=["community_area"]).copy()

    # Derive fatality/injury approximations
    crimes["is_homicide"] = crimes["primary_type"].astype(str).str.upper().eq("HOMICIDE")
    desc_upper = crimes["description"].astype(str).str.upper()
    crimes["is_injury"] = crimes["primary_type"].astype(str).str.upper().eq("BATTERY") & (
        desc_upper.str.contains("AGGRAVATED|SHOOT|SHOT|HANDGUN|FIREARM")
    )

    monthly = (
        crimes.groupby(["community_area", "month_start"]).agg(
            incident_count=("id", "count"),
            arrests=("arrest", "sum"),
            domestic=("domestic", "sum"),
            fatalities=("is_homicide", "sum"),
            injuries=("is_injury", "sum"),
        )
        .reset_index()
    )

    socio = pd.read_csv(RAW_SOCIO_FILE)
    socio = _standardize_columns(socio)
    # Rename common fields
    if "community area number" in socio.columns:
        socio = socio.rename(columns={"community area number": "community_area"})
    if "community_area_number" in socio.columns:
        socio = socio.rename(columns={"community_area_number": "community_area"})

    socio["community_area"] = pd.to_numeric(socio["community_area"], errors="coerce")
    socio = socio.dropna(subset=["community_area"]).copy()

    # Merge without month (static socio indicators)
    agg = monthly.merge(socio, on="community_area", how="left")

    # Derive rate per capita using per_capita_income as proxy for affluence; actual population not available here
    # Create normalized z-scores for select socio factors
    for col in [
        "percent_households_below_poverty",
        "percent_aged_16_unemployed",
        "percent_aged_25_without_high_school_diploma",
        "percent_aged_under_18_or_over_64",
        "percent_of_housing_crowded",
        "per_capita_income",
        "hardship_index",
    ]:
        if col in agg.columns:
            vals = agg[col].astype(float)
            agg[f"z_{col}"] = (vals - vals.mean()) / (vals.std() + 1e-9)

    agg.to_csv(AGG_MONTHLY_FILE, index=False)
    return agg, socio

