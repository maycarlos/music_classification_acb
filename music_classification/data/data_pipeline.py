from typing import Optional

import pandas as pd

from ..utils.dotenv_loading import ENV_CONFIG
from .process_data import clean_dataset, data_split, define_pipeline, process_data


def run(
    scenario: str, *, genre: Optional[str] = None
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], list]:
    df_raw = pd.read_csv(ENV_CONFIG["DATA_FOLDER"], dtype_backend="pyarrow")

    df = clean_dataset(df_raw, scenario, genre)

    # Split the data
    X_train, X_test, y_train, y_test, input_features = data_split(
        df, needs_interpretation=True
    )

    # process the data
    # encode the target variable
    # removes the outliers
    # balances the dataset
    X_train, X_test, y_train, y_test, label_encoder = process_data(
        X_train, X_test, y_train, y_test
    )

    # define the pipeline
    pipeline = define_pipeline()
    X_train = pipeline.fit_transform(X_train, y_train)
    X_test = pipeline.transform(X_test)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    training_data = pd.concat([X_train, y_train], axis=1)
    testing_data = pd.concat([X_test, y_test], axis=1)

    data = [training_data, testing_data]
    helper_objects = [input_features, label_encoder, pipeline]

    return data, helper_objects
