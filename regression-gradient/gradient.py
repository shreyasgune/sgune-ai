#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

true_b = 1
true_w = 2
N = 100 # number of points we're going to be working with.

# Data Generation
np.random.seed(42) # to create same random numbers for reproducibility
x = np.random.rand(N,1)
epsilon = (0.1 * np.random.randn(N,1)) #  which draws samples from a normal distribution (of mean 0 and variance 1)

#model 
y = true_b + true_w * x + epsilon

# Split the data in to training set and validation set

# The split should always be the first thing you do—no preprocessing, no
# transformations; nothing happens before the split. That’s why we do this
# immediately after the synthetic data generation.

idx = np.arange(N) # array of indices from 0 to N-1
np.random.shuffle(idx) # just shuffles stuff in place

train_idx = idx[:int(N * 0.8)] #train 80%
val_idx = idx[int(N * 0.8):] #validation 20%


x_train = x[train_idx], y[train_idx]
x_val = x[val_idx], y[val_idx]

# x_train[0] = training features
# x_train[1] = training labels

# x_val[0] = validation features
# x_val[1] = validation labels

# Forward pass
np.random.seed(42)  # For reproducibility
b = np.random.randn(1)
w = np.random.randn(1)
yhat = b + w * x_train[0]  # yhat = b + w * x

## Loss Calculation
# loss is generated using mean squared error (MSE).
# We are using ALL data points, so this is BATCH gradient descent. How wrong is our model? That's the error!
error = (yhat - y[train_idx])
loss = (error ** 2).mean()
print(f"Initial loss: {loss:.4f}")

## Lets do the same for more values of b and w
# evenly spaced, 3 units below, 3 units above, 80 number of points
b_range = np.linspace(true_b - 3, true_b + 3, 80)
w_range = np.linspace(true_w - 3, true_w + 3, 80)
#meshgrid is a handy function that generates a grid of b and w, so basically it creates an matrix.

bs, ws = np.meshgrid(b_range, w_range)
# print(bs)
print(bs.shape, ws.shape)

## Using the new matrix of a range of b and w values::

dummy_x = x_train[0]
print("dummy_x shape:", np.shape(dummy_x))
dummy_yhat = bs + ws * dummy_x

# dummy_yhat.shape
# skipping for loop to do this and use np.apply.along_axis instead.

all_predictions = np.apply_along_axis(
    func1d = lambda x: bs + ws * x,
    axis = 1,
    arr = x_train,
)

print(all_predictions.shape)

def plot_data(x_val, true_w, true_b, title):

    x_val_x, x_val_y = x_val

    plt.figure(figsize=(8, 6))

    # Scatter plot for validation data
    plt.scatter(x_val_x, x_val_y, color='blue', label=title)

    # Plot the true line y = true_b + true_w * x
    x_line = np.linspace(0, 1, 100).reshape(-1, 1)
    y_line = true_b + true_w * x_line
    plt.plot(x_line, y_line, 'k--', label=f'True line (y = {true_b} + {true_w}x)')

    # Labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot_data(x_val, true_w, true_b, 'Validation Data')
# plot_data(x_train, true_w, true_b, 'Training Data')

# # we are doing this to get a first pass at how different our results are from the scatter plot, cuz we are using random variables.
# plot_data(x_train, b, w, "Forward Pass (b, w)")



# Batch, Mini-batch, and Stochastic Gradient Descent
# • If we use all points in the training set (n = N) to compute the
# loss, we are performing a batch gradient descent;
# • If we were to use a single point (n = 1) each time, it would be a
# stochastic gradient descent;
# • Anything else (n) in between 1 and N characterizes a mini-
# batch gradient descent;

