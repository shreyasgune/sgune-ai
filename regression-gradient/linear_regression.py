import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Lets generate some synthetic data
np.random.seed(42)

X = np.linspace(0,10,50) #creates synthetic data, of 50 
print(f"OG vector: {X}")

true_w, true_b = 2.5, 1.0 #this is what our model should arrive to

Y = true_w * X + true_b + np.random.normal(0,2,size=X.shape)

# Gradient descent
def compute_loss(w,b,X,Y):
    Y_predicted = w * X + b 
    return np.mean((Y - Y_predicted) ** 2)

def gradients(w,b,X,Y):
    Y_predicted = w * X + b
    dw = -2 * np.mean(X * (Y - Y_predicted)) # slope of w
    db = -2 * np.mean(Y - Y_predicted)
    return dw,db 

# Init Param set
w, b = np.random.randn(), np.random.randn() #we're choosing this random, because if we choose it as 0, our initial loss will always be bad, which we wanna avoid, to reduce the number of iterations we'd need

lr = 0.0005 #learning rate
max_iter = 1000 # wanna make sure we're not taking more than a thousand iterations

tol = 4.9 #0.000001 , which is our stoppage threshold. As soon as the norm of our vector goes below this, we stop cuz our model has converged.


history = [
    (
        w,b,compute_loss(w,b,X,Y)
    )
]
print(f"Initial w,b and loss is =: {history}")

#lets iterate
for i in range(max_iter):
    dw, db = gradients(w,b,X,Y)
    # For debugging
    # print(f"Normal(size) of slope: {np.linalg.norm([dw,db])}")
    if np.linalg.norm([dw,db]) < tol:
        print(f"We hit convergence at iteration: {i}")
        break 
    w = w - (dw * lr)
    b = b - (db * lr)
    loss = compute_loss(w,b,X,Y)
    history.append((w,b,loss))
    # We gotta stop as soon as our model shows sign of convergence
    # We need to measures how big the gradient is.
    # If the gradient is large → we’re far from the minimum.
    # If the gradient is close to zero → we’re near a flat region / minimum.tol is a small threshold (like 1e-6), so:
    # If the gradient length < tol, it means updates are negligible → gradient descent has effectively converged → we stop early.

    # For Reference:
    # np.linalg.norm() → vector/matrix norm (length/magnitude).
    # np.linalg.inv() → matrix inverse.
    # np.linalg.det() → determinant.
    # np.linalg.eig() → eigenvalues & eigenvectors.

#if the slope of the loss surface is smaller than this, further updates won’t change the parameters in any meaningful way. Taking the norm gives the size of the gradient vector (how steep the slope still is).

print(f"Length of history: {len(history)}")
start_w, start_b, start_loss = history[0]
mid_w, mid_b, mid_loss = history[len(history)//2]
final_w, final_b, final_loss = history[-1]

print(f"Converged after {len(history)} iterations")
print(f"Best fit line : y = {final_w:.3f}x + {final_b:.3f}")
print(f"where w={final_w:.3f} and b={final_b:.3f}")


# Plot the Loss
# Extract the loss values from history
_, _, hist_loss = zip(*history)

plt.figure(figsize=(8,6))
plt.plot(range(len(hist_loss)), hist_loss, color="purple")
plt.title("Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()


#Model Data Points
plt.figure(figsize=(8,6))
plt.scatter(X,Y,label="Data Points")

#Model at the start
plt.plot(X, start_w * X + start_b, color="blue", linestyle="--", label="start model")

#Model at the mid-point
plt.plot(X, mid_w * X + mid_b, color="green", linestyle="dashdot", label="mid model")

# Model line at the end
plt.plot(X, final_w * X + final_b, color="red", label="Final Line")

plt.title("Model Line at Start vs End of Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#plot the data points and the best-fit line we came up with
#Model Data Points
plt.figure(figsize=(8,6))
plt.scatter(X,Y,label="Data Points")
plt.plot(X, final_w * X + final_b, color="red", label = "Best Fit Line")
plt.title("Linear Regression with Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


#plot the loss surface
w_vals = np.linspace(final_w - 2, final_w + 2, 50)
b_vals = np.linspace(final_b - 2, final_b + 2, 50)

#that linspace returns 50, evenly spaced numbers, from a to b, (to greate a grid, of w,b in our case) spanning +- 2 units (arbitrarily taken). 50 is resolution, higher the number, smoother is the surface, but will need more compute.

W, B = np.meshgrid(w_vals, b_vals)
#this turns 1-D array into 2-D matrix so we can calculate loss at every (w,b) pair.
#Shape of W,B is (50,50)

# Example
# w_vals = [0,1,2]
# b_vals = [10,11,12]
# 
# W = [[0,1,2],
    #  [0,1,2],
    #  [0,1,2]]
# 
# B = [[10,10,10],
    #  [11,11,11],
    #  [12,12,12]]
# 

Loss = np.array(
    [
        [
            compute_loss(w,b,X,Y) #MSE calculation based on  y = wx+b
            for w in w_vals # iterates weights across columns for that bias 
        ]
        for b in b_vals #create one row per bias value
    ]
)

# Faster, vector method of doing this
# Wg, Bg = np.meshgrid(w_vals, b_vals)                # shape (B, W)
# # make predictions for every (w,b) across all X at once:
# Y_pred = Wg[:, :, None] * X[None, None, :] + Bg[:, :, None]  # shape (B, W, N)
# Loss = ((Y_pred - Y[None, None, :]) ** 2).mean(axis=2)       # shape (B, W)

# The resulting shape of the loss is same as B and W.
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111,projection="3d")
ax.plot_surface(W,B,Loss,cmap="viridis", alpha=0.8) # alpha makes the surface slightly transparent, viridis controls colours

# Path of gradient descent
hist_w, hist_b, hist_loss = zip(*history) # this unpacks history tuple into those three vars
ax.plot(hist_w, hist_b, hist_loss, color="red", marker="o", markersize=2, label="Descent Path")

ax.set_xlabel("Weight (w)")
ax.set_ylabel("Bias (b)")
ax.set_zlabel("Loss")
ax.set_title("Loss Surface and Gradient Descent Path")
ax.legend()
plt.show()
