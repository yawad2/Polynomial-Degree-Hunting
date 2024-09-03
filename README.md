# Polynomial Degree Hunting

## Author
Yousra Awad  
Contact: yawad2@u.rochester.edu

## Project Overview
This project implements and compares regularized gradient descent and regularized least squares regression for polynomial fitting. The model automatically selects the weights that minimize the error.

## Key Features
- Implements both regularized gradient descent and regularized least squares regression
- Automatically determines optimal polynomial degree
- Compares performance of both methods on clean and noisy datasets

## Optimal Parameters
- Gradient Descent:
  - Learning rate: 0.01
  - Regularization constant (lambda): 0.01
  - Number of steps: 5000
- Least Squares Regression:
  - Regularization constant: 0.001

## Usage
Run the program from the command line, providing training and test data files as arguments:
