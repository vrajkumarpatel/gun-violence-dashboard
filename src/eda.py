import os
from typing import Dict

import pandas as pd
import plotly.express as px

from .config import (
    CLEANED_CPD_FILE,
    AGG_MONTHLY_FILE,
    RAW_COMMUNITY_AREAS_GEOJSON,
    REPORTS_DIR,
)


def ensure_reports_dir() -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    figs_dir = os.path.join(REPORTS_DIR, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    return figs_dir


def time_series_trends() -> str:
    figs_dir = ensure_reports_dir()
    df = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"])
    df = df[df["gun_related"]]
    monthly = df.groupby("month_start").size().rename("count").reset_index()
    fig = px.line(monthly, x="month_start", y="count", title="Monthly Gun-Related Incidents")
    out = os.path.join(figs_dir, "time_series_monthly.html")
    fig.write_html(out)
    return out


def incidents_distribution() -> Dict[str, str]:
    figs_dir = ensure_reports_dir()
    df = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"])
    df = df[df["gun_related"]]

    by_type = df.groupby("primary_type").size().rename("count").reset_index().sort_values("count", ascending=False).head(20)
    fig_type = px.bar(by_type, x="primary_type", y="count", title="Top Incident Types (Gun-Related)")
    out_type = os.path.join(figs_dir, "distribution_types.html")
    fig_type.write_html(out_type)

    by_hour = df.groupby("hour").size().rename("count").reset_index()
    fig_hour = px.bar(by_hour, x="hour", y="count", title="Incidents by Hour")
    out_hour = os.path.join(figs_dir, "distribution_hour.html")
    fig_hour.write_html(out_hour)

    by_weekday = df.groupby("day_of_week").size().rename("count").reset_index()
    fig_wd = px.bar(by_weekday, x="day_of_week", y="count", title="Incidents by Day of Week")
    out_wd = os.path.join(figs_dir, "distribution_weekday.html")
    fig_wd.write_html(out_wd)

    return {"types": out_type, "hour": out_hour, "weekday": out_wd}


def hotspot_map() -> str:
    figs_dir = ensure_reports_dir()
    df = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"])
    df = df[df["gun_related"]]
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        radius=10,
        center=dict(lat=41.8781, lon=-87.6298),
        zoom=9,
        mapbox_style="open-street-map",
        title="Gun Violence Hotspot Density (Chicago)",
    )
    out = os.path.join(figs_dir, "hotspot_density.html")
    fig.write_html(out)
    return out


def community_area_heatmap() -> str:
    figs_dir = ensure_reports_dir()
    agg = pd.read_csv(AGG_MONTHLY_FILE)
    # Total incidents per community
    totals = agg.groupby("community_area")["incident_count"].sum().reset_index()
    import json
    with open(RAW_COMMUNITY_AREAS_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)
    fig = px.choropleth(
        totals,
        geojson=gj,
        locations="community_area",
        featureidkey="properties.area_numbe",
        color="incident_count",
        color_continuous_scale="Reds",
        scope="usa",
        title="Total Gun-Related Incidents by Community Area",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    out = os.path.join(figs_dir, "community_area_heatmap.html")
    fig.write_html(out)
    return out
