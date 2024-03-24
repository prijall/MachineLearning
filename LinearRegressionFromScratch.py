import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate=learning_rate
        self.iterations= iterations
        self.weights= None 
        self.bias=None
    
    def fit(self, X, y):
        num_samples, num_features=X.shape
        self.weights=np.zeros(num_features)
        self.bias=0
        
        # Gradient Descent

        for _ in range(self.iterations):
            # predict th eoutput using the current weights and bias

            y_pred= self.predict(X)

            # Calculate the gradients

            dw=(1/num_samples) * np.dot(X.T, (y_pred-y))
            db= (1/num_samples) * np.sum(y_pred-y)

            #update the weights and bias

            self.weights = np.random.randn(1, X_train.shape[1])

            self.bias -= self.learning_rate * db


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data Generation
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3*X + noise

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate Linear Regression Model
model = LinearRegression(learning_rate=0.01, iterations=1000)

# Fit the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the Model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)