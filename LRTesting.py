from LogisticsRegressionFromScratch import LogisticRegression
import numpy as np


X= np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y=np.array([0, 0, 1, 1])

X_test=np.array([[1, 1], [5, 6]])

lr=LogisticRegression()
lr.fit(X, y)
predictions = lr.predict(X_test)
print("Predictions:", predictions)