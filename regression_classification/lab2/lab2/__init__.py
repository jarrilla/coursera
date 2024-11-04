from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from .utils import *
import copy
import math

def main():
    X_train, y_train = load_data()
    # check_data(X_train, y_train)

    test_w = 0.2
    test_b = 0.2

    w, b, J_history, w_history = gradient_descent(X_train, y_train, test_w, test_b, 0.01, 10000)

    m = X_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * X_train[i] + b

    plt.plot(X_train, predicted, c = "b")
    plt.scatter(X_train, y_train, c = "r", marker="x")
    plt.show()

    print(f"w: {w}, b: {b}")

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w, b. Updates w, b by taking 
    num_iters gradient steps with learning rate alpha
    """
    m = len(X)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # print(f"dj_dw: {dj_dw}, dj_db: {dj_db}, w: {w}, b: {b}")

        if i < 100000:
            J_history.append(compute_cost(X, y, w, b))
        
        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {J_history[-1]:8.2f}")
        
    return w, b, J_history, w_history

def compute_cost(X, y, w, b):
    """
    Compute the cost over all examples
    Args:
      X (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w, b (scalar)    : model parameters  
    Returns
      total_cost (float): The cost of the model function
    """
    m = X.shape[0]
    total_cost = 0

    for i in range(m):
        cost_i = (w * X[i] - y[i] + b) ** 2
        total_cost += cost_i
    total_cost /= 2 * m
    return np.float64(total_cost)

def compute_gradient(X, y, w, b):
    """
    Compute the gradient for linear regression 
    """
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * X[i] + b
        temp_djdw = (f_wb - y[i]) * X[i]
        temp_djdb = f_wb - y[i]

        dj_dw += temp_djdw
        dj_db += temp_djdb
    
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def check_data(X_train, y_train):
    plt.scatter(X_train, y_train, marker='x', c='r')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Training Data')
    plt.show()
