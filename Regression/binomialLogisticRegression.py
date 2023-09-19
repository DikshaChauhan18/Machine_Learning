"""
LOGISTIC REGRESSION : used for binary classification problems
                    : where the dependent variable (target) is categorical and
                     has only two possible outcomes 0 &1.
                    : this algo is used in ml, statistics etc;
                    :eg pam email detection (spam or not spam),
                     disease diagnosis (diseased or healthy),
                    : goal is to find coffient
                    : eq =>  P(Y=1|X) = 1 / (1 + e^(-z))

                    where, 
                        e: This is the base of the natural logarithm, approximately equal to 2.71828.
                         P(Y=1|X) prob of y at given x.............y=dependent x independent.

                        z = b0 + b1*X1 + b2*X2 + ... + bn*Xn ==>z is combi of x, linear predictor

                    types of logistic regression are :  1)binomial
                                                        2)multinomal
                                                        3)Ordinal Logistic Regression
"""





#BINOMIAL LOGISTIC REGRESSION :statistical method used for binary classification problems
                            # the dependent variablehas only two possible categories, often labeled as 0 & 1. 
                            # P(Y=1|X) = 1 / (1 + e^(-z))
                            # z = b0 + b1*X1 + b2*X2 + ... + bn*Xn


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate some synthetic data for binary classification
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#test_size=0.3 specifies that 30% of the data will be used for testing, and 70% for training.

# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)  #Use the trained model to make predictions on the test data.

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
