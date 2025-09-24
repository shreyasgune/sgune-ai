import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from plotting import scatter_plot, contour_plot, ddd_plot, line_plot, plotter

# Now we see how scaling our data badly can have horrifying effects.

true_b = 1
true_w = 2
N = 100

np.random.seed(42)

bad_w = true_w / 10
bad_x = np.random.rand(N,1) * 10
y = true_b + bad_w * bad_x + (0.1 * np.random.randn(N,1))
print(y)

idx = np.arange(N) # array of indices from 0 to N-1
np.random.shuffle(idx) # just shuffles stuff in place
train_idx = idx[:int(N * 0.8)] #train 80%
val_idx = idx[int(N * 0.8):] #validation 20%
bad_x_train, y_train = bad_x[train_idx], y[train_idx]
bad_x_val, y_val = bad_x[val_idx], y[val_idx]
