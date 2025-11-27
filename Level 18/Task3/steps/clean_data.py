from typing import Tuple
import pandas as pd
import numpy as np
from zenml import step
from typing_extensions import Annotated


@step
def clean_data(
    X: np.ndarray, 
    y: np.ndarray
) -> Tuple[
    Annotated[pd.DataFrame, "cleaned_features"],
    Annotated[np.ndarray, "targets"]
]:
    """
    Clean the data by:
    1. Converting to DataFrame
    2. Adding a dummy column with NaN values (to demonstrate cleaning)
    3. Dropping unnecessary columns
    4. Filling null values with median
    
    Args:
        X: Feature array
        y: Target array
        
    Returns:
        Cleaned DataFrame and target array
    """
    # Convert to DataFrame
    df = pd.DataFrame(X)
    
    # Add a dummy column with NaN values to demonstrate cleaning
    df["dummy_column"] = np.nan
    
    # Drop the dummy column (simulating removing unnecessary columns)
    df = df.drop(columns=["dummy_column"])
    
    # Fill any null values with median
    df = df.fillna(df.median(numeric_only=True))
    
    print(f"Data cleaned: {len(df)} rows, {len(df.columns)} columns")
    print(f"Null values after cleaning: {df.isnull().sum().sum()}")
    
    return df, y
