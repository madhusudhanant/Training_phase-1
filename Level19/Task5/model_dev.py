from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        mlflow.sklearn.autolog()  
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
