"""
LINEAR REGRESSION with single variable  :- unsupervied learning
                                        :-statical method
                                        :-use for predictive analysis
                                        :-prediction of continous ,real,numerical values
                                        :-eg cost,price,age,sales,temp etc.
                                        :- y=mx+c+error

"""




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (you can replace this with your own dataset)
# For example, you can load a dataset from a CSV file:
# df = pd.read_csv('your_dataset.csv')

# Create synthetic data for demonstration purposes
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Feature: Random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Target: Linear relationship with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize the results (for 1D data)
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='b', label='Actual')
plt.plot(X_test, y_pred, color='r', label='Predicted')
plt.legend()
plt.xlabel('Feature')
plt.ylabel('Price')
plt.show()
