import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def main():
    # ex_linear()
    ex_sigmoid()
    
def ex_linear():
    X_train = np.array([[1.0], [2.0]], dtype=np.float32)
    y_train = np.array([[300.0], [500.0]], dtype=np.float32)

    linear_layer = Dense(units=1, activation='linear')
    
    a1 = linear_layer(X_train[0].reshape(1, 1))
    # w, b = linear_layer.get_weights()
    set_w = np.array([[200]])
    set_b = np.array([100])

    linear_layer.set_weights([set_w, set_b])
    
    prediction_tf = linear_layer(X_train)
    plt.plot(X_train, prediction_tf, c='b', label='TF Prediction')
    plt.scatter(X_train, y_train, c='r', label='Real Data')
    plt.legend()
    plt.show()

def ex_sigmoid():
    X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
    Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

    model = Sequential([
        Dense(units=1, input_dim=1, activation='sigmoid', name='L1')
    ])
    model.summary()

    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    
    logistic_layer = model.get_layer('L1')
    logistic_layer.set_weights([set_w, set_b])

    predict_tf = model(X_train)

    plt.plot(X_train, predict_tf, c='b', label='TF Prediction')
    plt.scatter(X_train, Y_train, c='r', label='Real Data')
    plt.legend()
    plt.show()