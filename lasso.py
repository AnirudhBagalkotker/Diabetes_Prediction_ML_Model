import numpy as np

# Define Lasso regression function
def lasso_regression(X, y, alpha, num_iterations, learning_rate):
    m, n = X.shape
    theta = np.zeros(n)
    
    for iteration in range(num_iterations):
        # Calculate predictions
        y_pred = np.dot(X, theta)
        
        # Calculate the loss
        error = y_pred - y
        loss = np.mean(error ** 2) + alpha * np.sum(np.abs(theta))  # L1 regularization
        
        # Calculate the gradient
        gradient = (1/m) * np.dot(X.T, error) + alpha * np.sign(theta)  # L1 gradient
        
        # Update the weights (theta) using gradient descent
        theta -= learning_rate * gradient
    
    return theta

# Lasso regression hyperparameters
alpha = 0.01  # Regularization strength
num_iterations = 1000
learning_rate = 0.01

# Add a column of ones to the features for the intercept term
X_train = np.column_stack((np.ones(len(x_tr)), x_tr.values))

# Perform Lasso regression
lasso_theta = lasso_regression(X_train, y_tr.values, alpha, num_iterations, learning_rate)

# Make predictions on the test data
X_test = np.column_stack((np.ones(len(x_te)), x_te.values))
y_pred = np.dot(X_test, lasso_theta)

# Calculate Mean Squared Error (MSE) and R-squared (R2) for evaluation
mse = mean_squared_error(y_te, y_pred)
r2 = r2_score(y_te, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Print the Lasso coefficients
lasso_coefficients = dict(zip(['Intercept'] + list(x_tr.columns), lasso_theta))
print("Lasso Coefficients:")
for feature, coef in lasso_coefficients.items():
    print(f"{feature}: {coef}")
