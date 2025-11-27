train_model.py

from zenml.steps import step

@step(experiment_tracker="mlflow_tracker")
def train_model(X_train, y_train):
    from model_dev import LinearRegressionModel
    model = LinearRegressionModel()
    return model.train(X_train, y_train)
