# Data-Driven Insights into Gun Violence for Community Intervention

## Live App

Open the deployed Streamlit dashboard:

https://gun-violence-dashboard-l3zedtcamrmhu2yx7ekbcv.streamlit.app/

## Screenshots

![Dashboard KPIs](Screenshot%202025-11-28%20024849.png)
![Filters and Charts](Screenshot%202025-11-28%20024807.png)
![Choropleth Map](Screenshot%202025-11-28%20024653.png)

This project builds a complete analytics system to analyze gun violence incidents in Chicago, identify trends and hotspots, predict high-risk neighborhoods, and provide interactive dashboards and reports for community-based intervention programs.

## Data Sources

- Chicago Police Department Crimes: `https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2` (programmatic access via Socrata API)
- Socioeconomic indicators by Chicago Community Area (2008–2012): `https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2`
- Community Areas boundaries (GeoJSON): `https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6`
- Gun Violence Archive (GVA) methodology and definitions: `https://www.gunviolencearchive.org/methodology`

Notes:
- CPD crime data are updated daily and may be subject to reclassification and errors. Addresses are block-level only.
- GVA collects incidents from 5,000+ sources daily; numbers can change as evidence is verified.

## Quick Start

1. Install dependencies:

```
python -m pip install -r requirements.txt
```

2. Run end-to-end pipeline (fetch, clean, EDA, model, report):

```
python main.py --fetch
```

3. Launch the dashboard:

```
python main.py --run-dashboard
```

Then open `http://127.0.0.1:8050/` in your browser.

## Outputs

- Cleaned CPD data: `data/processed/cpd_crimes_clean.parquet`
- Monthly community metrics: `data/processed/community_monthly_metrics.parquet`
- Risk ranking: `data/processed/risk_ranking.csv`
- Figures (HTML): `reports/figures/*`
- Model metrics: `reports/model_metrics.json`
- Final report: `reports/final_report.md`

## Approach

- Ingestion: Pulls CPD crimes (last ~365 days by default), socioeconomic indicators (community areas), and community boundaries.
- Preprocessing: Standardizes columns, derives time features, flags gun-related incidents using description and type heuristics, aggregates monthly by community area, merges socioeconomic features.
- EDA: Time series trends, incident distributions, hotspot density map, and community area heatmap.
- Modeling: Logistic regression to classify high-risk community-months (top quartile by incident counts). Outputs neighborhood risk scores.
- Dashboard: Interactive filters for date, type, and community areas with KPIs and visualizations.

## Updating Datasets

- Re-run `python main.py --fetch` to refresh with the latest crimes.
- The socioeconomic indicators are static (2008–2012). You can swap in a newer dataset by replacing the CSV URL in `src/config.py` and ensuring column names align.

## Data Literacy

- Interpret results alongside socioeconomic context; the hardship index and related indicators are proxies and may not capture recent changes.
- Use the dashboard to explore temporal patterns (e.g., evenings/weekends) and spatial hotspots to inform targeted interventions.
