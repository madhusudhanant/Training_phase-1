from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true, y_pred):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class R2(Evaluation):
    def calculate_scores(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse_eval = MSE()
    r2_eval = R2()

    print("MSE:", mse_eval.calculate_scores(y_test, y_pred))
    print("R2:", r2_eval.calculate_scores(y_test, y_pred))
