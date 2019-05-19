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
    ignored_regions = [
        "Tunisia",
        "North East",
        "Center East",
        "Center West",
        "South East",
        "South West",
        "North West",
    ]
    data = (
        pd.read_csv(input_filepath)
        .pipe(
            lambda df: df.assign(
                regions=df.regions.str.replace("Governorate of", "")
                .str.replace("du", "")
                .str.strip()
            )
        )
        .pipe(lambda df: df.loc[(-df["regions"].isin(ignored_regions))])
        .pipe(lambda df: df.drop(columns=["Unit"]))
        .pipe(lambda df: df.rename(columns=lambda x: x.lower()))
        .pipe(
            lambda df: df.rename(
                columns={"regions": "region", "indicators": "indicator"}
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
