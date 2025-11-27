import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from zenml import step


@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    """
    Train a Linear Regression model on the input data.
    
    Args:
        df: Input pandas DataFrame with features and target
        
    Returns:
        Trained LinearRegression model
    """
    print("Starting model training...")
    
    # Split features and target
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column as target
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
   
    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model training complete.")
    print(f"R² on training data: {train_score:.4f}")
    print(f"R² on test data: {test_score:.4f}")
    
    return model
