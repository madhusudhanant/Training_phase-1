from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from sklearn.datasets import load_digits


@step
def load_digits_data() -> Tuple[
    Annotated[np.ndarray, "features"],
    Annotated[np.ndarray, "targets"]
]:
    """Step that loads the digits dataset from scikit-learn."""
    digits = load_digits()
    return digits.data, digits.target
