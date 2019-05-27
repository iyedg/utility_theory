# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Python [conda env:utility_theory] *
#     language: python
#     name: conda-env-utility_theory-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd

# %%
drop_cols = [
    "Indicators: Key",
    "Indicators: Last Update",
    "Indicators: Sources",
    "Indicators: Comment",
    "Regions: ISO",
    "Scale: Key",
    "Indicators: Methodology",
    "Scale: Name",
    "Frequency",
    "Indicators: Unit",
    "Regions: Key",
]
col_names = {
    "Facts: Value": "value",
    "Indicators: Full name": "indicator",
    "Regions: Name": "governorate",
    "Date": "year",
}
indicator_names = {
    "Total number of doctors": "Number of doctors",
    "Total number of paramedical staff": "Number of paramedical staff",
    "The number of beds for the public health sector": "Number of beds in the public health sector",
    "The total number of pharmacies": "Number of pharmacies",
    "Estimate of the population on July 1st": "Population",
}

# %%
df = (
    pd.read_excel(
        r"D:\School\Advanced decision theory\utility_theory_tunisia_health\data\raw\Socioeconomic 05_26_2019 04_50_03.xls"
    )
    .drop(columns=drop_cols)
    .rename(columns=col_names)
    .pipe(
        lambda df: df.assign(
            year=df["year"].apply(pd.to_datetime).apply(lambda x: x.year)
        )
    )
    .pipe(lambda df: df.loc[(df["year"] == 2015)])
    .drop(columns=["year"])
    .pipe(lambda df: df.assign(value=df["value"].apply(pd.to_numeric)))
    .pipe(
        lambda df: df.assign(
            governorate=df["governorate"].apply(
                lambda val: val.replace("Governorate of", "").replace("du ", "").strip()
            )
        )
    )
    .pipe(
        lambda df: df.loc[
            df.indicator.isin(
                [
                    "Total number of doctors",
                    "Total number of paramedical staff",
                    "Number of public basic health centers",
                    "The number of beds for the public health sector",
                    "The total number of pharmacies",
                    "Estimate of the population on July 1st",
                    "Number of institutes, centers and specialized hospitals in public sector",
                    "Number of public district hospitals",
                    "Number of regional public hospitals",
                    "The number of public hospitals",
                ]
            )
        ]
    )
    .replace(indicator_names)
    .pivot(index="governorate", columns="indicator", values="value")
    .pipe(lambda df: df.apply(lambda col: col / df["Population"] * 1000, axis=0))
)

# %% [markdown]
# ## All values are calculated as Number per 1000 inhabitants

# %%
df.to_csv(
    r"D:\School\Advanced decision theory\utility_theory_tunisia_health\data\processed\health_infrastructure.csv"
)

# %%
