from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on given data"""
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train):
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        return reg


if __name__ == "__main__":
  
    data = load_diabetes()
    X = data.data
    y = data.target

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegressionModel()
    trained_model = model.train(X_train, y_train)


    print("Trained model coefficients:", trained_model.coef_)
