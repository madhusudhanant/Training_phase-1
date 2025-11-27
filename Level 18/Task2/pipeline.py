from zenml import pipeline, step
from sklearn import datasets
import numpy as np
from typing import Tuple
from typing_extensions import Annotated


@step
def load_digits_data() -> Tuple[
    Annotated[np.ndarray, "features"],
    Annotated[np.ndarray, "targets"]
]:
    """Step that loads the digits dataset from scikit-learn."""
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target classes: {np.unique(y)}")
    
    return X, y


@pipeline
def simple_data_pipeline():
    """A simple pipeline that loads the digits dataset."""
    X, y = load_digits_data()
    return X, y


if __name__ == "__main__":
    print("Starting ZenML pipeline execution...")
    pipeline_run = simple_data_pipeline()
    print("Pipeline execution completed!")
    print(f"Pipeline run ID: {pipeline_run.id}")
