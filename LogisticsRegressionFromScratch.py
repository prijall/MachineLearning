import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.w=None
        self.b=0

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    

    def compute_cost(self, y, y_pred):
        m=len(y)
        return -1/m * np.sum(y*np.log(y_pred)+ (1-y)*np.log(1-y_pred))
    
    def fit(self, X, y):
        m, n=X.shape
        self.w=np.zeros(n)
        self.bias=0

        for i in range(self.epochs):
            y_pred=self.sigmoid(np.dot(X, self.w)+self.b)
            gradient=np.dot(X.T, (y_pred-y))/m
            self.w-=self.learning_rate * gradient
            self.b-=self.learning_rate * np.sum(y_pred-y)/m

            cost=self.compute_cost(y, y_pred)
            if i % 1000==0:
                print(f'Iteration {i}: Cost= {cost: 4f}')

    
    def predict(self, X):
        y_pred=self.sigmoid(np.dot(X, self.w)+ self.b)
        return (y_pred>0.5).astype(int)