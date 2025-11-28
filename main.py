import argparse
import os
import pandas as pd

from src.data_ingestion import fetch_cpd_crimes, fetch_socioeconomic, fetch_community_areas_geojson
from src.preprocessing import load_and_clean, build_monthly_by_community
from src.eda import time_series_trends, incidents_distribution, hotspot_map, community_area_heatmap
from src.modeling import train_and_evaluate
from src.report import generate_final_report
from src.config import REPORTS_DIR


def run_pipeline(days: int = 365, start: str = None, end: str = None) -> None:
    if start and end:
        fetch_cpd_crimes(start_date=start, end_date=end)
    else:
        fetch_cpd_crimes()
    fetch_socioeconomic()
    fetch_community_areas_geojson()

    load_and_clean()
    build_monthly_by_community()

    time_series_trends()
    incidents_distribution()
    hotspot_map()
    community_area_heatmap()

    train_and_evaluate()
    generate_final_report()
    print(f"Pipeline complete. See reports in: {REPORTS_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="Gun Violence Analytics Pipeline")
    p.add_argument("--fetch", action="store_true", help="Fetch latest data")
    p.add_argument("--start", type=str, default=None, help="ISO start date")
    p.add_argument("--end", type=str, default=None, help="ISO end date")
    p.add_argument("--run-dashboard", action="store_true", help="Run Dash app")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fetch:
        run_pipeline(start=args.start, end=args.end)
    if args.run_dashboard:
        from dashboard.app import main as dash_main
        dash_main()


if __name__ == "__main__":
    main()

