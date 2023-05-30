from typing import Optional

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.dotenv_loading import ENV_CONFIG
from ..utils.types_ import DataFrame

SEED = int(ENV_CONFIG["SEED"])


def clean_dataset(
    raw_data: DataFrame,
    scenario: Optional[str] = "B",
    target_genre: Optional[str] = None,
):
    """

    Args:
        df_raw (_type_): _description_

    Returns:
        _type_: _description_
    """

    processed_data = (
        raw_data.pipe(convert_numeric)
        .pipe(convert_category, scenario, target_genre)
        .drop(["filename"], axis=1)
        .rename(columns={"label": "genre"})
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    return processed_data


def convert_numeric(dataframe: DataFrame):
    """
    Convert all the numerical features type to a float with 32 bits
    """

    dataframe = dataframe.copy()
    dataframe[dataframe.select_dtypes("number").columns] = dataframe.select_dtypes(
        "number"
    ).astype(np.float32)
    return dataframe


def convert_category(
    dataframe: DataFrame,
    scenario: Optional[str] = None,
    target_genre: Optional[str] = "blues",
):
    """
    Converts the only categorical features to the pandas datatype "category" according to the project scenario of interest

    Args
    ----
        dataframe: Dara to be transformed.
        scenario: Scenario consired. Defaults to None.
    """

    dataframe = dataframe.copy()

    if scenario == "A":
        dataframe["label"] = dataframe["label"].where(
            dataframe["label"] == target_genre, "others"
        )

    dataframe["label"] = dataframe["label"].astype("category")

    return dataframe


def data_split(dataframe: DataFrame, needs_interpretation: bool = False):
    # * Data Splitting
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values
    input_features = None

    if needs_interpretation:
        input_features = dataframe.columns[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=SEED, test_size=0.2
    )

    return X_train, X_test, y_train, y_test, input_features


def process_data(X_train, X_test, y_train, y_test):
    # * Encode Target Variable
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # * Deal with outliers
    lof = LocalOutlierFactor()
    yhet = lof.fit_predict(X_train)
    mask = yhet != -1
    X_train, y_train = X_train[mask, :], y_train[mask]

    # * Balance classes for training
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, le


def define_pipeline():
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("variance_threshold", VarianceThreshold(threshold=0.25)),
            ("k_best", SelectKBest(mutual_info_classif, k=25)),
            ("pca", PCA(n_components=0.95)),
        ]
    ).set_output(transform="pandas")

    return pipeline
