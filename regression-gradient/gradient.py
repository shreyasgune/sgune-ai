#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plotting import scatter_plot, contour_plot, ddd_plot, line_plot

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


x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# x_train[0] = training features
# x_train[1] = training labels

# x_val[0] = validation features
# x_val[1] = validation labels


print("To train a model, we need to randomly initialize the parameters/weights (b and w)")

# Forward pass
np.random.seed(42)  # For reproducibility
b = np.random.randn(1)
w = np.random.randn(1)
print("Random b and w:",b,w)

yhat = b + w * x_train  # yhat = b + w * x
print("Computing our models' predicted output using yhat equation:",yhat.shape)

## Loss Calculation
# loss is generated using mean squared error (MSE).
# We are using ALL data points, so this is BATCH gradient descent. How wrong is our model? That's the error!
error = (yhat - y_train)
loss = (error ** 2).mean()
print(f"Initial loss: {loss:.4f}")

# We have just computed the loss (2.74) corresponding to our randomly initialized
# parameters (b = 0.49 and w = -0.13).

## Lets do the same for more values of b and w
# evenly spaced, 3 units below, 3 units above, 80 number of points
b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)

#meshgrid is a handy function that generates a grid of b and w, so basically it creates an matrix.
bs, ws = np.meshgrid(b_range, w_range)
# print(bs)
print(bs.shape, ws.shape)

## Using the new matrix of a range of b and w values::

# lets take a single data point from the training set(x_train) and compute predictions for every combination in our grid
print("single point from x_train set:", x_train[0])
dummy_x = x_train[0]

#lets do a dummy prediction using all the points in the bs and ws mesh of size (101,101)
dummy_yhat = bs + ws * dummy_x

print("dummy_yhat now contains predictions and is of shape(101,101)",dummy_yhat.shape)


# we want to multiply the same x value by every entry in the ws matrix. This operation resulted
# in a grid of predictions for that single data point. Now we need to do this for every
# one of our 80 data points in the training set.

print("Calculating all predictions from the 80 data points in the training set.")
all_predictions = np.apply_along_axis(
    func1d = lambda x: bs + ws * x,
    axis = 1,
    arr = x_train,
)

print("We got 80 matrices of shape (101, 101), one matrix for each data point, each matrix containing a grid of predictions::",all_predictions.shape)


# The errors would be the difference between the our predicted values and the labels(y). Before we do that though, we need to reshape it so it can be workable.
all_labels = y_train.reshape(-1,1,1)
# reshape is a method in NumPy used to change the shape of an array without changing its data.

# -1 means: "infer this dimension automatically based on the other dimensions and the total number of elements."
print("y_train reshaped to ", all_labels.shape)

#Now we can take the MSE. Take square of all errors, then average the squares over all data points. Set axis=0 cuz our data points are in the first dimension.

all_errors = (all_predictions - all_labels)
print("All errors using 80 matrices of shape(101,101)::", all_errors.shape)
all_losses = (all_errors ** 2).mean(axis=0)
print("All losses of shape", all_losses.shape)



# for the absolute majority of problems, computing the loss surface is not going to be feasible: we have to rely on gradient descent’s ability to reach a point of minimum, even if it starts at some random point


# Lets now compute the gradients.
b_gradient = 2 * error.mean()
w_gradient = 2 * (x_train * error).mean()
print("Gradient for b and w:",b_gradient,w_gradient)



# Batch, Mini-batch, and Stochastic Gradient Descent
# • If we use all points in the training set (n = N) to compute the
# loss, we are performing a batch gradient descent;
# • If we were to use a single point (n = 1) each time, it would be a
# stochastic gradient descent;
# • Anything else (n) in between 1 and N characterizes a mini-
# batch gradient descent;

print("===========")
print("Learning Rate - eta")
lr = 0.1
print(b,w)
b = b - lr * b_gradient
w = w - lr * w_gradient
print(b,w)







#--------------PLOTTING-------------#
# line_plot(b,w,[-b_gradient,-w_gradient],"b","w","Gradient")
# scatter_plot(x_train, b, w, "Forward Pass (b, w)")
# scatter_plot(x_val, true_w, true_b, 'Validation Data')
# scatter_plot(x_train, true_w, true_b, 'Training Data')
# contour_plot(bs,ws,"bs","ws","Loss Surface")
# ddd_plot(bs,ws,all_losses,"bs","ws","Losses","Loss Surface")