import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate, no_of_iterations, alpha):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.alpha = alpha

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights_and_bias()

    def update_weights_and_bias(self):
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)

        # Update the weights and bias using Ridge regularization
        self.w = self.w - self.learning_rate * (dw + 2 * self.alpha * self.w)
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred

    def get_coefficients(self):
        return self.w, self.b

# ... (previous code)

# Create and train the Ridge Regression model
ridge_regression = RidgeRegression(learning_rate=0.01, no_of_iterations=1000, alpha=0.01)
ridge_regression.fit(x_tr.values, y_tr.values)

# Get and print the Ridge coefficients
ridge_coefficients = ridge_regression.get_coefficients()
print("Ridge Coefficients:")
print("Coefficients (w):", ridge_coefficients[0])
print("Bias (b):", ridge_coefficients[1])

# Make predictions on the test data
y_pred = ridge_regression.predict(x_te.values)

# Calculate Mean Squared Error (MSE) manually
mse = np.mean((y_te.values - y_pred) ** 2)

# Calculate R-squared (R2) manually
y_mean = np.mean(y_te.values)
ss_total = np.sum((y_te.values - y_mean) ** 2)
ss_residual = np.sum((y_te.values - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Continue with the rest of your code (if any)
# ...
