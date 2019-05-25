import pandas as pd
from pathlib import Path


def load_data():

    project_dir = Path(__file__).resolve().parents[2]
    return pd.read_csv(
        Path(project_dir, "data/processed/health_infrastructure.csv")
    ).pivot_table(index="date", columns=["indicator", "region"], values="value")
