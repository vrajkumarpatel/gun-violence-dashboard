import os
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from src.config import (
    CLEANED_CPD_FILE,
    AGG_MONTHLY_FILE,
    RISK_RANKING_FILE,
    DASH_HOST,
    DASH_PORT,
)


def load_data():
    crimes = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"])
    crimes["gun_related"] = crimes["gun_related"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    agg = pd.read_csv(AGG_MONTHLY_FILE, parse_dates=["month_start"])
    ranking = pd.read_csv(RISK_RANKING_FILE)
    return crimes, agg, ranking


def build_app() -> Dash:
    crimes, agg, ranking = load_data()
    crimes = crimes[crimes["gun_related"]]

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Gun Violence Insights (Chicago)"

    neighborhoods = sorted(crimes["community_area"].dropna().astype(int).unique())
    incident_types = sorted(crimes["primary_type"].dropna().unique())

    # KPIs
    total_incidents = int(agg["incident_count"].sum())
    total_injuries = int(agg["injuries"].sum()) if "injuries" in agg.columns else 0
    total_fatalities = int(agg["fatalities"].sum()) if "fatalities" in agg.columns else 0
    total_arrests = int(agg["arrests"].sum()) if "arrests" in agg.columns else 0
    domestic_flags = int(agg["domestic"].sum()) if "domestic" in agg.columns else 0
    last_month = agg["month_start"].max()
    last_month_incidents = int(agg.loc[agg["month_start"] == last_month, "incident_count"].sum())

    kpi_layout = html.Div([
        html.Div(f"Total Incidents: {total_incidents}", className="kpi-card"),
        html.Div(f"Total Injuries: {total_injuries}", className="kpi-card"),
        html.Div(f"Total Fatalities: {total_fatalities}", className="kpi-card"),
        html.Div(f"Total Arrests: {total_arrests}", className="kpi-card"),
        html.Div(f"Domestic Incidents: {domestic_flags}", className="kpi-card"),
        html.Div(f"Last Month Incidents: {last_month_incidents}", className="kpi-card"),
    ], className="kpi-container")

    app.layout = dbc.Container([
        kpi_layout,
        html.H2("Data-Driven Insights into Gun Violence for Community Intervention"),
        html.P("Interactive dashboard: hotspots, trends, and risk scoring for Chicago."),
        dbc.Row([
            dbc.Col([
                html.Label("Date range (month start)"),
                dcc.DatePickerRange(
                    id="date-range",
                    min_date_allowed=crimes["month_start"].min(),
                    max_date_allowed=crimes["month_start"].max(),
                    start_date=crimes["month_start"].min(),
                    end_date=crimes["month_start"].max(),
                ),
            ], width=4),
            dbc.Col([
                html.Label("Incident type"),
                dcc.Dropdown(options=[{"label": t, "value": t} for t in incident_types], id="type", multi=True),
            ], width=4),
            dbc.Col([
                html.Label("Community area"),
                dcc.Dropdown(options=[{"label": str(n), "value": int(n)} for n in neighborhoods], id="ca", multi=True),
            ], width=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="time-series"), width=6),
            dbc.Col(dcc.Graph(id="hotspot"), width=6),
        ]),

        html.H4("Risk Ranking by Community Area (Logistic Regression)"),
        dcc.Graph(
            figure=px.bar(ranking, x="community_area", y="risk_score", title="Neighborhood Risk Scores (LR)"),
            id="ranking"
        ),

        html.H4("Community Risk Ranking (Random Forest)"),
        dcc.Graph(id="rf-ranking"),

        html.H3("Scenario Modeling: Reduce Incidents"),
        dcc.Slider(id="reduction-slider", min=0, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(0, 51, 5)}, tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id="scenario-chart"),
    ], fluid=True)

    @app.callback(
        Output("time-series", "figure"),
        Output("hotspot", "figure"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("type", "value"),
        Input("ca", "value"),
    )
    def update_figures(start_date, end_date, types, cas):
        dff = crimes.copy()
        if start_date:
            dff = dff[dff["month_start"] >= pd.to_datetime(start_date)]
        if end_date:
            dff = dff[dff["month_start"] <= pd.to_datetime(end_date)]
        if types:
            dff = dff[dff["primary_type"].isin(types)]
        if cas:
            dff = dff[dff["community_area"].astype(float).isin(cas)]

        monthly = dff.groupby("month_start").size().rename("count").reset_index()
        ts_fig = px.line(monthly, x="month_start", y="count", title="Monthly Incidents")

        dff = dff.dropna(subset=["latitude", "longitude"]).copy()
        dff["latitude"] = pd.to_numeric(dff["latitude"], errors="coerce")
        dff["longitude"] = pd.to_numeric(dff["longitude"], errors="coerce")
        dff = dff.dropna(subset=["latitude", "longitude"]).copy()
        hotspot_fig = px.density_mapbox(
            dff,
            lat="latitude",
            lon="longitude",
            radius=10,
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=9,
            mapbox_style="open-street-map",
            title="Hotspot Density",
        )
        return ts_fig, hotspot_fig

    @app.callback(
        Output("rf-ranking", "figure"),
        Output("scenario-chart", "figure"),
        Input("reduction-slider", "value"),
    )
    def update_rf_and_scenario(reduction_percent):
        # RF ranking using latest month
        agg_latest = agg[agg["month_start"] == agg["month_start"].max()].copy()
        if "rf_risk_score" in agg_latest.columns:
            fig_rf = px.bar(
                agg_latest.sort_values("rf_risk_score", ascending=False),
                x="community_area",
                y="rf_risk_score",
                color="rf_risk_score",
                color_continuous_scale="Reds",
                title="Community Risk Ranking (Random Forest)",
            )
        else:
            fig_rf = px.bar(
                agg_latest.sort_values("incident_count", ascending=False),
                x="community_area",
                y="incident_count",
                title="Community Ranking by Incidents (No RF)",
            )

        df_sc = agg_latest.copy()
        df_sc["adjusted_incidents"] = df_sc["incident_count"] * (1 - (reduction_percent or 0) / 100.0)
        # simple adjusted risk normalized 0-1
        m = df_sc["adjusted_incidents"].max() or 1.0
        df_sc["adjusted_risk"] = df_sc["adjusted_incidents"] / m
        fig_scenario = px.bar(
            df_sc,
            x="community_area",
            y="adjusted_risk",
            color="adjusted_risk",
            color_continuous_scale="Blues",
            title=f"Projected Risk after {reduction_percent}% Incident Reduction",
        )
        return fig_rf, fig_scenario

    return app


def main():
    app = build_app()
    app.run_server(host=DASH_HOST, port=DASH_PORT, debug=False)


if __name__ == "__main__":
    main()
