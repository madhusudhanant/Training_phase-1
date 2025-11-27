import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from zenml import step
from typing import Dict


@step
def evaluate_model(
    model: LinearRegression,
    data: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained LinearRegression model
        data: Input DataFrame with features and target
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print("Starting model evaluation...")
    
    # Split features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2_score': float(r2)
    }
    
    print(f"Evaluation complete:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    return metrics
