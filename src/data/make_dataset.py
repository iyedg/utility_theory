# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    data = (
        pd.read_excel(input_filepath)
        .pipe(
            lambda df: df.drop(
                columns=[
                    "Indicators: Sources",
                    "Indicators: Unit",
                    "Indicators: Methodology",
                    "Indicators: Comment",
                    "Indicators: Key",
                    "Regions: Key",
                    "Regions: ISO",
                    "Scale: Key",
                    "Scale: Name",
                    "Frequency",
                ]
            )
        )
        .pipe(
            lambda df: df.assign(
                **{
                    "Indicators: Last Update": df["Indicators: Last Update"].apply(
                        pd.to_datetime
                    ),
                    "Date": df["Date"].apply(pd.to_datetime),
                }
            )
        )
        .pipe(lambda df: df.loc[(df["Date"] < df["Indicators: Last Update"].min())])
        .pipe(lambda df: df.assign(**{"Date": df["Date"].apply(lambda x: x.year)}))
        .pipe(lambda df: df.drop(columns=["Indicators: Last Update"]))
        .pipe(
            lambda df: df.rename(
                columns={
                    "Facts: Value": "value",
                    "Indicators: Full name": "indicator",
                    "Regions: Name": "region",
                    "Date": "date",
                }
            )
        )
    )
    data.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
