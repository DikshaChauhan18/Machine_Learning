"""
LINEAR REGRESSION WITH MULTIPLE VALUES
                data prediction is dependent on multiple values
                :-eq = Y = b0 + b1*X1 + b2*X2 + b3*X3 + ... + bn*Xn + ε
                :- more than 1 independent var == X1,X,2,X3.....
                :- use to to find the values of the coefficients (b0, b1, b2, ..., bn)
                 that result in the best fit to the observed data.
                 :-using ordinary least squares (OLS) regression
                 :- ols=  often referred to simply as "linear regression," 
                 the sum of squared differences between the predicted and actual values is minimized.

                 :-"linear regression" refers to the general class of models used to describe
                  linear relationships between variables, "OLS regression" specifically refers
                   to the method for estimating the coefficients of a linear regression model. 

"""
# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file.
# Your dataset should have multiple features (independent variables) and a target variable (dependent variable).
# For this example, we'll use a synthetic dataset.
np.random.seed(0)
X = np.random.rand(100, 3)  # Generate random data for three features
y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + np.random.randn(100)  # Create a linear relationship with some noise
"""
y represents the target variable, which is the variable we want to predict or explain.
X[:, 0], X[:, 1], and X[:, 2] represent the values of three independent variables. 
"""
# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a linear regression model
model = LinearRegression()

# Step 5: Train the model on the training data
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue', label='Data Points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal')
plt.title('Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()




"""
Blue Data Points: These are the actual vs. predicted data points obtained from your regression model. 
Each point represents the actual value (on the x-axis) and the corresponding predicted value
 (on the y-axis) for a specific data sample.

Red Dashed Line (Ideal Line): This diagonal line represents the ideal scenario where the actual and 
predicted values are equal. It serves as a reference line to compare how well your model's
predictions align with the actual values. In the ideal scenario, all data points would fall exactly
on this line.
"""
