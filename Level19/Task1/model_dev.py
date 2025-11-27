from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on given data"""
        pass

if __name__ == "__main__":
    try:
        model = Model() 
    except TypeError as e:
        print(f"Error: {e}")
