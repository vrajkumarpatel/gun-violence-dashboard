"""
Configuration constants for data sources and application settings.
"""

# Chicago Police Department Crimes dataset (Socrata API)
CPD_CRIMES_ENDPOINT = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

# Socioeconomic indicators by Community Area (CSV)
SOCIOECONOMIC_CSV_URL = (
    "https://data.cityofchicago.org/api/views/kn9c-c2s2/rows.csv?accessType=DOWNLOAD"
)

# Community Areas boundaries GeoJSON
COMMUNITY_AREAS_GEOJSON_URL = (
    "https://data.cityofchicago.org/resource/cauq-8yn6.geojson"
)

# Default data paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "reports"

# Files
RAW_CPD_FILE = f"{RAW_DIR}/cpd_crimes_raw.csv"
RAW_SOCIO_FILE = f"{RAW_DIR}/socioeconomic_indicators.csv"
RAW_COMMUNITY_AREAS_GEOJSON = f"{RAW_DIR}/community_areas.geojson"

CLEANED_CPD_FILE = f"{PROCESSED_DIR}/cpd_crimes_clean.csv"
AGG_MONTHLY_FILE = f"{PROCESSED_DIR}/community_monthly_metrics.csv"
RISK_RANKING_FILE = f"{PROCESSED_DIR}/risk_ranking.csv"
MODEL_METRICS_FILE = f"{REPORTS_DIR}/model_metrics.json"

# Gun-related keyword heuristics for CPD descriptions
GUN_KEYWORDS = [
    "HANDGUN",
    "FIREARM",
    "RIFLE",
    "REVOLVER",
    "GUN",
    "SHOT",
    "SHOTS",
    "WEAPON",
]

# Dashboard config
DASH_PORT = 8050
DASH_HOST = "127.0.0.1"
