# Project-wide path and ticker constants.
from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"
OUTPUTS_MODELS = ROOT / "outputs" / "models"

TICKER = "WM"
COMPANY_NAME = "Waste Management Inc"
FY_END_MONTH = 12  # December 31
START_DATE = "2022-07-01"
END_DATE = "2026-03-31"
