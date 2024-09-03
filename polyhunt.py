import numpy as np
import sys
np.random.seed(42)  # Set seed for reproducibility

# Load data from text file
def load_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

# Create design matrix with polynomial features up to a maximum degree
def create_design_matrix(x, degree):
    X = np.column_stack([x**i for i in range(degree + 1)])
    return X

# Regularized Gradient Descent
def regularized_gradient_descent(X, y, lmbda, learning_rate, iterations):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y) + (lmbda/m) * theta
        theta -= learning_rate * gradients
    return theta

# Regularized Least Squares Regression
def regularized_least_squares(X, y, lmbda):
    m, n = X.shape
    L = lmbda * np.eye(n)
    theta = np.linalg.inv(X.T.dot(X) + L).dot(X.T).dot(y)
    return theta

# Print predicted polynomial
def print_polynomial(theta):
    degree = len(theta) - 1  # Degree of the polynomial
    polynomial = f"y = {theta[-1]:.6f}X^{degree}"  # Initialize polynomial string with highest degree term

    # Add terms for each degree of the polynomial in descending order
    for i in range(degree - 1, -1, -1):
        if theta[i] != 0:  # Only include non-zero coefficients
            if i == 0:
                polynomial += f" + {theta[i]:.6f}"
            elif i == 1:
                polynomial += f" + {theta[i]:.6f}X"
            else:
                polynomial += f" + {theta[i]:.6f}X^{i}"

    print("Predicted Polynomial:", polynomial)

# Train the model, use cross validation, and find the degree that minimizes the error
def find_degree(x, y, k_fold=5):
    # Initialize variables
    max_degree = 11  # Maximum degree for polynomial features
    best_mse = float('inf')
    best_degree = 0
    best_theta_gd = None
    best_theta_ls = None
    
    # Split data into k folds
    fold_size = len(x) // k_fold
    indices = np.random.permutation(len(x))
    x_folds = [x[indices[i*fold_size:(i+1)*fold_size]] for i in range(k_fold)]
    y_folds = [y[indices[i*fold_size:(i+1)*fold_size]] for i in range(k_fold)]
    
    # Iterate over increasing degrees of polynomial features
    for degree in range(1, max_degree + 1):
        mse_sum = 0
        for i in range(k_fold):
            x_train = np.concatenate([x_folds[j] for j in range(k_fold) if j != i])
            y_train = np.concatenate([y_folds[j] for j in range(k_fold) if j != i])
            x_val = x_folds[i]
            y_val = y_folds[i]
            
            # Create design matrix with polynomial features
            X_train = create_design_matrix(x_train, degree)
            X_val = create_design_matrix(x_val, degree)

            # Regularized Gradient Descent
            theta_gd = regularized_gradient_descent(X_train, y_train.reshape(-1, 1), lmbda=0.1, learning_rate=0.01, iterations=5000)

            # Regularized Least Squares Regression
            theta_ls = regularized_least_squares(X_train, y_train, lmbda=0.001)

            # Calculate Mean Squared Error on validation set for both methods
            mse_gd = np.mean((X_val.dot(theta_gd) - y_val.reshape(-1, 1))**2)
            mse_ls = np.mean((X_val.dot(theta_ls) - y_val)**2)
            
            mse_sum += min(mse_gd, mse_ls)

        avg_mse = mse_sum / k_fold
        
        # Update best degree if current model has lower mse
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_theta_gd = theta_gd
            best_theta_ls = theta_ls

    return best_degree, best_theta_gd, best_theta_ls

def main():
    # Get data file from terminal
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # Load data from files
    x, y = load_data(train_file)
    x_test, y_test = load_data(test_file)


    #lambda
    lmda = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # Find best degree and theta
    best_degree, best_theta_gd, best_theta_ls = find_degree(x, y)

    # Predict on test data and calculate error
    x_test_poly = create_design_matrix(x_test, best_degree)
    mse_gd = np.mean((x_test_poly.dot(best_theta_gd) - y_test.reshape(-1, 1))**2)
    mse_ls = np.mean((x_test_poly.dot(best_theta_ls) - y_test)**2)

    # Round the parameters for readability and flatten
    best_theta_gd_rounded = [round(x, 6) for x in np.flip(best_theta_gd.flatten()).tolist()]
    best_theta_ls_rounded = [round(x, 6) for x in np.flip(best_theta_ls.flatten()).tolist()]
   
    # Print results
    print("Gradient Descent: ")
    print("Theta (GD): ", best_theta_gd_rounded) # Weights in ascending order for gradient descent
    print_polynomial(best_theta_gd.flatten())
    print("Error on testing data: ", mse_gd)
    print("")
    print("Least Square Regression: ")
    print("Theta (LSR): ", best_theta_ls_rounded) # Weights in ascending order for least squares regression
    print_polynomial(best_theta_ls.flatten())
    print("Error on testing data: ", mse_ls)
    print("")
    print("Predicted Degree:", best_degree)

if __name__ == "__main__":
    main()
