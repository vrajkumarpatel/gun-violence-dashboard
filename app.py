import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests

# Import project configuration paths
from src.config import (
    CLEANED_CPD_FILE,
    AGG_MONTHLY_FILE,
    RISK_RANKING_FILE,
    RAW_COMMUNITY_AREAS_GEOJSON,
    CPD_CRIMES_ENDPOINT,
    SOCIOECONOMIC_CSV_URL,
    COMMUNITY_AREAS_GEOJSON_URL,
)

def _detect_fid_key(geojson):
    try:
        feats = geojson.get("features", [])
        if not feats:
            return "properties.area_numbe"
        props = feats[0].get("properties", {})
        for candidate in [
            "area_numbe",
            "area_num",
            "community",
            "area_number",
        ]:
            if candidate in props:
                return f"properties.{candidate}"
        return "properties.area_numbe"
    except Exception:
        return "properties.area_numbe"

# -----------------------------
# Data loading helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    crimes = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"]) if os.path.exists(CLEANED_CPD_FILE) else pd.DataFrame()
    agg = pd.read_csv(AGG_MONTHLY_FILE, parse_dates=["month_start"]) if os.path.exists(AGG_MONTHLY_FILE) else pd.DataFrame()
    ranking_lr = pd.read_csv(RISK_RANKING_FILE) if os.path.exists(RISK_RANKING_FILE) else pd.DataFrame()
    geojson = None
    try:
        import json as _json
        if os.path.exists(RAW_COMMUNITY_AREAS_GEOJSON):
            with open(RAW_COMMUNITY_AREAS_GEOJSON, "r", encoding="utf-8") as f:
                geojson = _json.load(f)
    except Exception:
        geojson = None

    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df["month_start"] = df["date"].dt.to_period("M").dt.start_time
        return df

    def _gun_flag(df: pd.DataFrame) -> pd.DataFrame:
        desc = df.get("description", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
        prim = df.get("primary_type", pd.Series(index=df.index, dtype=str)).astype(str).str.upper()
        keywords = ["HANDGUN", "FIREARM", "RIFLE", "REVOLVER", "GUN", "SHOT", "SHOTS", "WEAPON"]
        kw = desc.apply(lambda s: any(k in s for k in keywords))
        ph = prim.str.contains("WEAPONS VIOLATION|HOMICIDE|BATTERY|ASSAULT|ROBBERY", regex=True)
        df["gun_related"] = kw | ph
        return df

    if crimes.empty:
        try:
            # First attempt: Socrata JSON API (no $select to avoid parameter compatibility issues)
            params = {"$limit": 20000}
            resp = requests.get(CPD_CRIMES_ENDPOINT, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                crimes = pd.DataFrame(data)
            else:
                crimes = pd.DataFrame()
        except Exception:
            crimes = pd.DataFrame()

        # Fallback: CSV bulk download if JSON API failed or returned empty
        if crimes.empty:
            try:
                csv_url = "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD"
                crimes = pd.read_csv(csv_url)
            except Exception:
                crimes = pd.DataFrame()

        if not crimes.empty:
            crimes = _standardize_columns(crimes)
            # Ensure expected columns exist
            if "id" not in crimes.columns and "case_number" in crimes.columns:
                crimes = crimes.rename(columns={"case_number": "id"})
            crimes = _parse_dates(crimes)
            crimes = _gun_flag(crimes)
            for c in ["arrest", "domestic"]:
                if c in crimes.columns:
                    crimes[c] = crimes[c].astype(str).str.lower().map({"true": True, "false": False, "t": True, "f": False})

    if agg.empty:
        if crimes.empty:
            agg = pd.DataFrame()
        else:
            # Align monthly aggregation with gun-related scope when available
            crimes_scoped = crimes.copy()
            if "gun_related" in crimes_scoped.columns:
                crimes_scoped = crimes_scoped[crimes_scoped["gun_related"]]
            crimes_ca = crimes_scoped.dropna(subset=["community_area"]).copy()
            crimes_ca["community_area"] = pd.to_numeric(crimes_ca["community_area"], errors="coerce")
            crimes_ca = crimes_ca.dropna(subset=["community_area"]).copy()
            crimes_ca["is_homicide"] = crimes_ca["primary_type"].astype(str).str.upper().eq("HOMICIDE")
            desc_upper = crimes_ca["description"].astype(str).str.upper()
            crimes_ca["is_injury"] = crimes_ca["primary_type"].astype(str).str.upper().eq("BATTERY") & (
                desc_upper.str.contains("AGGRAVATED|SHOOT|SHOT|HANDGUN|FIREARM")
            )
            monthly = (
                crimes_ca.groupby(["community_area", "month_start"]).agg(
                    incident_count=("id", "count"),
                    arrests=("arrest", "sum"),
                    domestic=("domestic", "sum"),
                    fatalities=("is_homicide", "sum"),
                    injuries=("is_injury", "sum"),
                ).reset_index()
            )
            try:
                socio = pd.read_csv(SOCIOECONOMIC_CSV_URL)
                socio = _standardize_columns(socio)
                if "community area number" in socio.columns:
                    socio = socio.rename(columns={"community area number": "community_area"})
                if "community_area_number" in socio.columns:
                    socio = socio.rename(columns={"community_area_number": "community_area"})
                socio["community_area"] = pd.to_numeric(socio["community_area"], errors="coerce")
                socio = socio.dropna(subset=["community_area"]).copy()
                agg = monthly.merge(socio, on="community_area", how="left")
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
            except Exception:
                agg = monthly

    if ranking_lr.empty and not agg.empty:
        df = agg.copy()
        thr = df["incident_count"].quantile(0.75)
        df["high_risk"] = (df["incident_count"] >= thr).astype(int)
        df_sorted = df.sort_values(["community_area", "month_start"])  
        latest = df_sorted.drop_duplicates(subset=["community_area"], keep="last").copy()
        m = latest["incident_count"].max() or 1.0
        latest.loc[:, "risk_score"] = latest["incident_count"] / m
        ranking_lr = latest[["community_area", "risk_score", "incident_count"]]

    if geojson is None:
        try:
            gj_resp = requests.get(COMMUNITY_AREAS_GEOJSON_URL, timeout=60)
            gj_resp.raise_for_status()
            geojson = gj_resp.json()
        except Exception:
            geojson = None

    return crimes, agg, ranking_lr, geojson


# -----------------------------
# Sidebar filters
# -----------------------------
def sidebar_filters(crimes: pd.DataFrame, agg: pd.DataFrame):
    st.sidebar.header("Filters")
    if crimes.empty and agg.empty:
        # Always return 5 items even when data is missing
        today = pd.Timestamp.today()
        return (today, today), [], [], 0, "Geo (offline)"

    months_series = None
    if not agg.empty and "month_start" in agg.columns:
        months_series = agg["month_start"].dropna().sort_values().unique()
    elif not crimes.empty and "month_start" in crimes.columns:
        months_series = crimes["month_start"].dropna().sort_values().unique()
    else:
        months_series = np.array([pd.Timestamp.today()])
    min_ts = pd.to_datetime(months_series.min())
    max_ts = pd.to_datetime(months_series.max())
    start_def = max_ts - pd.DateOffset(months=11)
    if start_def < min_ts:
        start_def = min_ts
    min_d = min_ts.date()
    max_d = max_ts.date()
    dr = st.sidebar.date_input("Date range", value=(start_def.date(), max_d))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = dr
    else:
        start = dr if not isinstance(dr, (list, tuple)) else min_d
        end = dr if not isinstance(dr, (list, tuple)) else max_d

    # Incident types from crimes
    types = sorted(crimes["primary_type"].dropna().unique()) if (not crimes.empty and "primary_type" in crimes.columns) else []
    type_sel = st.sidebar.multiselect("Incident types", options=types, default=[])

    # Community areas
    communities_crimes = crimes.get("community_area", pd.Series(dtype=float)) if not crimes.empty else pd.Series(dtype=float)
    communities_agg = agg.get("community_area", pd.Series(dtype=float)) if not agg.empty else pd.Series(dtype=float)
    ca_cr = pd.to_numeric(communities_crimes, errors="coerce").dropna().astype(int)
    ca_ag = pd.to_numeric(communities_agg, errors="coerce").dropna().astype(int)
    communities_vals = np.union1d(ca_cr.values, ca_ag.values)
    communities = sorted(communities_vals.tolist()) if communities_vals.size > 0 else []
    ca_sel = st.sidebar.multiselect("Community areas", options=communities, default=[])

    # Map mode and scenario modeling slider
    map_mode = st.sidebar.selectbox("Map mode", options=["Tile (OSM)", "Geo (offline)"], index=0)
    reduction = st.sidebar.slider("Incident reduction (%)", min_value=0, max_value=50, step=5, value=0)
    start_m = pd.to_datetime(start).to_period("M").start_time
    end_m = pd.to_datetime(end).to_period("M").end_time
    return (start_m, end_m), type_sel, ca_sel, reduction, map_mode


# -----------------------------
# KPI cards
# -----------------------------
def kpi_section(agg_filtered: pd.DataFrame):
    st.subheader("Key Metrics")
    if agg_filtered.empty:
        st.info("No data available for current filters")
        return
    total_incidents = int(agg_filtered["incident_count"].sum())
    total_injuries = int(agg_filtered.get("injuries", pd.Series(dtype=int)).sum())
    total_fatalities = int(agg_filtered.get("fatalities", pd.Series(dtype=int)).sum())
    total_arrests = int(agg_filtered.get("arrests", pd.Series(dtype=int)).sum())
    domestic_flags = int(agg_filtered.get("domestic", pd.Series(dtype=int)).sum())
    last_month = agg_filtered["month_start"].max()
    last_month_incidents = int(agg_filtered.loc[agg_filtered["month_start"] == last_month, "incident_count"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Incidents", f"{total_incidents:,}")
    c2.metric("Total Injuries", f"{total_injuries:,}")
    c3.metric("Total Fatalities", f"{total_fatalities:,}")
    c4.metric("Total Arrests", f"{total_arrests:,}")
    c5.metric("Domestic Incidents", f"{domestic_flags:,}")
    c6.metric("Last Month Incidents", f"{last_month_incidents:,}")


# -----------------------------
# Visualizations
# -----------------------------
def time_series(crimes_filtered: pd.DataFrame):
    st.subheader("Monthly Gun-Related Incidents")
    if crimes_filtered.empty:
        st.info("No data available for current filters")
        return
    monthly = crimes_filtered.groupby("month_start").size().rename("count").reset_index()
    fig = px.line(monthly, x="month_start", y="count")
    st.plotly_chart(fig, use_container_width=True)


def hotspot_map(crimes_filtered: pd.DataFrame, mode: str = "Tile (OSM)"):
    st.subheader("Hotspot Density (Chicago)")
    if crimes_filtered.empty:
        st.info("No data available for current filters")
        return
    df = crimes_filtered.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if mode == "Tile (OSM)":
        fig = px.density_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            radius=10,
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=9,
            mapbox_style="open-street-map",
        )
    else:
        fig = px.scatter_geo(
            df,
            lat="latitude",
            lon="longitude",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(title="Hotspot (Geo projection)")
    st.plotly_chart(fig, use_container_width=True)


def community_choropleth(agg_filtered: pd.DataFrame, geojson):
    st.subheader("Incidents by Community Area (Choropleth)")
    if agg_filtered.empty or geojson is None:
        st.info("No data available or boundary file missing")
        return
    totals = agg_filtered.groupby("community_area")["incident_count"].sum().reset_index()
    fid = _detect_fid_key(geojson)
    fig = px.choropleth(
        totals,
        geojson=geojson,
        locations="community_area",
        featureidkey=fid,
        color="incident_count",
        color_continuous_scale="Reds",
        scope="usa",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)


def ranking_charts(ranking_lr: pd.DataFrame, agg_filtered: pd.DataFrame):
    st.subheader("Risk Rankings")
    col1, col2 = st.columns(2)
    if not ranking_lr.empty:
        fig_lr = px.bar(ranking_lr.sort_values("risk_score", ascending=False), x="community_area", y="risk_score")
        col1.plotly_chart(fig_lr, use_container_width=True)
    else:
        col1.info("Logistic Regression ranking not available")

    latest = None
    if not agg_filtered.empty and "rf_risk_score" in agg_filtered.columns:
        latest_month = agg_filtered["month_start"].max()
        latest = agg_filtered[agg_filtered["month_start"] == latest_month].copy()
    if latest is not None and not latest.empty and "rf_risk_score" in latest.columns:
        fig_rf = px.bar(latest.sort_values("rf_risk_score", ascending=False), x="community_area", y="rf_risk_score", color="rf_risk_score", color_continuous_scale="Reds")
        col2.plotly_chart(fig_rf, use_container_width=True)
    else:
        col2.info("Random Forest ranking not available")


def scenario_modeling(agg_latest: pd.DataFrame, reduction_percent: int):
    st.subheader("Scenario Modeling: Reduce Incidents")
    if agg_latest.empty:
        st.info("No data available for current filters")
        return
    df_sc = agg_latest.copy()
    df_sc["adjusted_incidents"] = df_sc["incident_count"] * (1 - (reduction_percent or 0) / 100.0)
    m = df_sc["adjusted_incidents"].max() or 1.0
    df_sc["adjusted_risk"] = df_sc["adjusted_incidents"] / m
    fig = px.bar(df_sc, x="community_area", y="adjusted_risk", color="adjusted_risk", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Final report section
# -----------------------------
def final_report(ranking_lr: pd.DataFrame):
    st.subheader("Summary and Recommendations")
    try:
        with open(os.path.join("reports", "model_metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None

    if metrics:
        st.markdown(f"- Logistic Regression Accuracy: {metrics['accuracy']:.3f}")
        st.markdown(f"- Precision (macro): {metrics['macro avg']['precision']:.3f}")
        st.markdown(f"- Recall (macro): {metrics['macro avg']['recall']:.3f}")
        st.markdown(f"- F1 (macro): {metrics['macro avg']['f1-score']:.3f}")
    if not ranking_lr.empty:
        top5 = ranking_lr.sort_values("risk_score", ascending=False).head(5)
        st.markdown("**Top 5 High-Risk Community Areas (LR):**")
        for _, r in top5.iterrows():
            st.markdown(f"- Community Area {int(r['community_area'])}: risk_score={r['risk_score']:.3f}, incidents={int(r['incident_count'])}")

    st.markdown("- Focus outreach in top-risk community areas.")
    st.markdown("- Prioritize evening/weekend programming based on temporal patterns.")
    st.markdown("- Coordinate with local orgs to address socioeconomic drivers (poverty, unemployment, education).")


# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Gun Violence Data Dashboard for Community Intervention", layout="wide")
    st.title("Gun Violence Data Dashboard for Community Intervention")
    st.caption("Interactive KPIs, trends, maps, risk rankings, and scenario modeling for Chicago.")

    crimes, agg, ranking_lr, geojson = load_data()
    # Filter to gun-related crimes only if column present
    if not crimes.empty and "gun_related" in crimes.columns:
        crimes = crimes[crimes["gun_related"]]

    filters = sidebar_filters(crimes, agg)
    (start_date, end_date), type_sel, ca_sel, reduction, map_mode = filters

    # Apply filters on crimes
    crimes_f = crimes.copy()
    if not crimes_f.empty:
        crimes_f = crimes_f[(crimes_f["month_start"] >= pd.to_datetime(start_date)) & (crimes_f["month_start"] <= pd.to_datetime(end_date))]
        if type_sel:
            crimes_f = crimes_f[crimes_f["primary_type"].isin(type_sel)]
        if ca_sel:
            crimes_f["community_area"] = pd.to_numeric(crimes_f["community_area"], errors="coerce")
            crimes_f = crimes_f[crimes_f["community_area"].isin(ca_sel)]

    # Apply filters on monthly agg
    agg_f = agg.copy()
    if not agg_f.empty:
        agg_f = agg_f[(agg_f["month_start"] >= pd.to_datetime(start_date)) & (agg_f["month_start"] <= pd.to_datetime(end_date))]
        if ca_sel:
            agg_f["community_area"] = pd.to_numeric(agg_f["community_area"], errors="coerce")
            agg_f = agg_f[agg_f["community_area"].isin(ca_sel)]

    kpi_section(agg_f)
    time_series(crimes_f)
    hotspot_map(crimes_f, map_mode)
    community_choropleth(agg_f, geojson)
    ranking_charts(ranking_lr, agg_f)

    # Scenario modeling uses latest month subset
    if not agg_f.empty:
        latest_month = agg_f["month_start"].max()
        agg_latest = agg_f[agg_f["month_start"] == latest_month].copy()
    else:
        agg_latest = pd.DataFrame()
    scenario_modeling(agg_latest, reduction)
    final_report(ranking_lr)


if __name__ == "__main__":
    main()
