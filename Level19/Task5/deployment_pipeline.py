from zenml import step, pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.artifacts import Output
from model_dev import LinearRegressionModel
from evaluation import MSE, R2
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

@step
def train_model(X_train: Output, y_train: Output) -> LinearRegressionModel:
    model = LinearRegressionModel()
    return model.train(X_train, y_train)

@step
def evaluate_model(model: LinearRegressionModel, X_test: Output, y_test: Output) -> tuple:
    mse = MSE()
    r2 = R2()
    y_pred = model.predict(X_test)
    mse_score = mse.calculate_scores(y_test, y_pred)
    r2_score = r2.calculate_scores(y_test, y_pred)
    return mse_score, r2_score

@step
def deployment_trigger(r2_score: float) -> bool:
    return r2_score >= 0.5

@pipeline
def continuous_deployment_pipeline(X_train: Output, y_train: Output, X_test: Output, y_test: Output):
    model = train_model(X_train, y_train)
    mse_score, r2_score = evaluate_model(model, X_test, y_test)
    deploy = deployment_trigger(r2_score)
    if deploy:
        mlflow_model_deployer_step(model=model)

if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    continuous_deployment_pipeline(X_train, y_train, X_test, y_test)
