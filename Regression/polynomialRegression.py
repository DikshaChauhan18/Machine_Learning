"""
POLYNOMIAL REGRESSION :- it is used when we have to show non linear relationship between 1
 independent and dependent variable.
                        :- it gives more flexible and curved relationships between the variables.
                        :-general eq = y = b0 + b1*x + b2*x^2 + ... + bn*x^n + ε
                        -> y= dependent x= independent var
                        ->b0,b1,b2 =  are the coefficients of the polynomial terms
                        ->n= degree of poly
                        ->ε = error, accounting for random noise.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Create sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Create polynomial features
degree = 2  # Degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Fit a linear regression model to the transformed features
model = LinearRegression()
model.fit(X_poly, y)

# Predict values using the trained model
y_pred = model.predict(X_poly)

# Plot the original data and the polynomial regression curve
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Polynomial Regression (Degree {degree})')
plt.show()
