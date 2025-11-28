import os
import json
import pandas as pd

from .config import (
    CLEANED_CPD_FILE,
    RISK_RANKING_FILE,
    MODEL_METRICS_FILE,
    REPORTS_DIR,
)
from .modeling import train_random_forest


def generate_final_report() -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "final_report.md")

    crimes = pd.read_csv(CLEANED_CPD_FILE, parse_dates=["date", "month_start"])
    crimes = crimes[crimes["gun_related"]]
    total_incidents = len(crimes)
    arrests = crimes["arrest"].sum() if "arrest" in crimes.columns else 0
    domestic = crimes["domestic"].sum() if "domestic" in crimes.columns else 0
    latest_month = crimes["month_start"].max()
    monthly_counts = crimes.groupby("month_start").size().rename("count").reset_index()
    recent_trend = monthly_counts.tail(6)

    ranking = pd.read_csv(RISK_RANKING_FILE)
    top5 = ranking.sort_values("risk_score", ascending=False).head(5)

    with open(MODEL_METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    rf_metrics = train_random_forest()

    lines = []
    lines.append("# Data-Driven Insights into Gun Violence for Community Intervention")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append(f"- Total gun-related incidents analyzed: {total_incidents}")
    lines.append(f"- Arrests associated: {int(arrests)}")
    lines.append(f"- Domestic-related incidents: {int(domestic)}")
    lines.append(f"- Latest month covered: {latest_month.date()}")
    lines.append("")
    lines.append("## Recent 6-Month Trend (Incidents)")
    for _, row in recent_trend.iterrows():
        lines.append(f"- {row['month_start'].date()}: {int(row['count'])}")
    lines.append("")
    lines.append("## High-Risk Neighborhoods (Top 5 by risk score)")
    for _, r in top5.iterrows():
        lines.append(f"- Community Area {int(r['community_area'])}: risk_score={r['risk_score']:.3f}, incidents={int(r['incident_count'])}")
    lines.append("")
    lines.append("## Model Performance (Logistic Regression)")
    lines.append(f"- Accuracy: {metrics['accuracy']:.3f}")
    lines.append(f"- Precision (macro): {metrics['macro avg']['precision']:.3f}")
    lines.append(f"- Recall (macro): {metrics['macro avg']['recall']:.3f}")
    lines.append(f"- F1 (macro): {metrics['macro avg']['f1-score']:.3f}")
    lines.append("")
    lines.append("## Model Performance (Random Forest)")
    if rf_metrics.get("available"):
        lines.append(f"- Accuracy: {rf_metrics['accuracy']:.3f}")
    else:
        lines.append("- Random Forest not available (dependency missing)")
    lines.append("")
    lines.append("## Recommendations for Community Intervention")
    lines.append("- Focus outreach and resources in top-risk community areas identified.")
    lines.append("- Prioritize evening and weekend programming based on time-of-day and weekday patterns.")
    lines.append("- Coordinate with local organizations to address socioeconomic drivers (poverty, unemployment, education).")
    lines.append("- Track arrest and domestic-related indicators to tailor supportive services.")
    lines.append("")
    lines.append("## How to Interpret Visualizations")
    lines.append("- Time Series: Shows monthly fluctuations and seasonal patterns in incidents.")
    lines.append("- Hotspot Map: Highlights spatial concentration of incidents across Chicago.")
    lines.append("- Community Area Heatmap: Aggregates totals by official community boundaries.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
