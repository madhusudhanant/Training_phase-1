
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true, y_pred):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mlflow.log_metric("mse", mse)
        return mse

class R2(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mlflow.log_metric("r2", r2)
        return r2
