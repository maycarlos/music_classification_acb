from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

Array = TypeVar("Array", np.ndarray,Any)
DataFrame = TypeVar("DataFrame", pd.DataFrame,Any)
Model = TypeVar("Model", BaseEstimator, Any)