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


#### PLOTTING
# Plot
plt.figure(figsize=(8, 6))

# Scatter plot for training data
# plt.scatter(x_train[0], x_train[1], color='blue', label='Training data')

# Scatter plot for validation data
plt.scatter(x_val[0], x_val[1], color='red', label='Validation data')

# Plot the true line y = 1 + 2x (without noise)
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = true_b + true_w * x_line
plt.plot(x_line, y_line, 'k--', label='True line (y = 1 + 2x)')

# Labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Train/Validation Split with True Linear Relationship')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

