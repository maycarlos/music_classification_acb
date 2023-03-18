# -*- coding: utf-8 -*-
import logging
import pathlib

import click
import numpy as np
import pandas as pd

from ..utils.dotenv_loading import ENV_CONFIG
from ..utils.types_ import DataFrame

SEED = int(ENV_CONFIG["SEED"])


def clean_dataset(raw_data, scenario):
    """

    Args:
        df_raw (_type_): _description_

    Returns:
        _type_: _description_
    """

    processed_data = (
        raw_data
        .pipe(convert_numeric)
        .pipe(convert_category, scenario)
        .drop(["filename"], axis = 1)
        .rename(columns = {"label":"genre"})
        .sample(frac = 1, random_state = SEED)
        .reset_index(drop = True)
    )

    return processed_data


def convert_numeric(dataframe):
    """
    Convert all the numerical features type to a float with 32 bits
    """
    
    dataframe = dataframe.copy()
    dataframe[dataframe.select_dtypes("number").columns] = dataframe.select_dtypes("number").astype(np.float32)
    return dataframe
    
def convert_category(dataframe, scenario = None):
    """
    Converts the only categorical features to the pandas datatype "category" according to the project scenario of interest

    Args
    ----
        Datadra
        scenario (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    dataframe = dataframe.copy()

    if scenario == "A":
        
        dataframe["label"] = (
            dataframe["label"]
            .where(dataframe["label"] == "blues", "others")
        )

    dataframe["label"] = dataframe["label"].astype("category")

    return dataframe


# class MusicDataset(pd.DataFrame):
#     """
#     _summary_

#     Args
#     ----
#         pd (_type_): _description_
#     """
#     def __init__(self, data_file: pathlib.Path) -> None:
#         super().__init__()
#         self.data = pd.read_csv(data_file)

#     def clean(self) -> DataFrame:
#         clean_data = (
#             self.data
#             .pipe(self._convert_numeric)
#             .pipe(self._convert_category)
#             .drop(["filename"], axis=1)
#             .rename(columns={"label": "genre"})
#             .sample(frac=1)
#             .reset_index(drop=True)
#         )

#         return clean_data

#     def _convert_numeric(self, data: DataFrame) -> DataFrame:
#         data = data.copy()
#         data[data.select_dtypes("number").columns] = data.select_dtypes(
#             "number"
#         ).astype(np.float32)

#         return data

#     def _convert_category(self, data: DataFrame) -> DataFrame:
#         data = data.copy()
#         data["label"] = data["label"].astype("category")

#         return data


# @click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
# def main(input_filepath, output_filepath):
#     """Runs data processing scripts to turn raw data from (../raw) into
#     cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("making final data set from raw data")


# if __name__ == "__main__":
#     log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = pathlib.Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()
