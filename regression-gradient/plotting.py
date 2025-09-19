import matplotlib.pyplot as plt
import numpy as np

def contour_plot(X,Y,xlabel,ylabel,title):
    contour = plt.contour(X, Y, levels=50, cmap='viridis')
    plt.colorbar(contour,label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotter(y_line,scats,N,x_label,y_label,title):
    x_line = np.linspace(0,1,N).reshape(-1,1)
    plt.figure(figsize=(8,6))
    val_x, val_y = scats
    plt.scatter(val_x,val_y, color='blue', label=title)
    plt.plot(x_line, y_line, 'k--', label=title)

    # Labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def ddd_plot(X,Y,Z,xlabel,ylabel,zlabel,title):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()    

def scatter_plot(x_val, true_w, true_b, title):

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

def line_plot(x,y,point,xlabel,ylabel,title):
    plt.plot(x,y,label=title,color='blue')
    if point:
        x_point=float(point[0])
        y_point=float(point[1])
        # plt.scatter(x_point, y_point, color='red', s=100, zorder=5, label=f'Point ({x_point}, {y_point})')
        plt.arrow(float(x),float(y),x_point,y_point,color='red', head_width=0.05, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
