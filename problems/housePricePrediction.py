import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data for demonstration purposes
np.random.seed(0)
num_samples = 100
X = np.random.rand(num_samples, 1) * 10  # House size (in square feet)
y = 2 * X + 3 + np.random.randn(num_samples, 1)  # House price (in 1000s of dollars)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the training data and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regression Line')
plt.title('House Price Prediction')
plt.xlabel('House Size (Square Feet)')
plt.ylabel('Price (in $1000s)')
plt.legend()
plt.show()

# Predict the price of a new house
new_house_size = np.array([[7.5]])  # Replace with the size of the new house
predicted_price = model.predict(new_house_size)
print(f"Predicted Price for New House: ${predicted_price[0][0] * 1000:.2f}")