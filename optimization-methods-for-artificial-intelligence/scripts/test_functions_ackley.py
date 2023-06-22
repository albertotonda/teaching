# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 08:31:36 2023

@author: Alberto
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ackley_function(x, global_optimum=None):
    if isinstance(x, np.ndarray):
        n = len(x)
        sum_sq_term = np.sum(x**2)
        cos_term = np.sum(np.cos(2*np.pi*x))
    else:
        n = 1
        sum_sq_term = x**2
        cos_term = np.cos(2*np.pi*x)

    term1 = -0.2 * np.sqrt(sum_sq_term / n)
    term2 = cos_term / n
    result = -20 * np.exp(term1) - np.exp(term2) + 20 + np.exp(1)

    if global_optimum is not None:
        result += global_optimum

    return result

    

# Generate input values for x, y, and calculate z
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X = np.zeros((x.shape[0] * y.shape[0],))
Y = np.zeros((x.shape[0] * y.shape[0],))
Z = np.zeros((x.shape[0] * y.shape[0],))

for i in range(0, x.shape[0]) :
    for j in range(0, y.shape[0]) :
        index = i*y.shape[0] + j
        X[index] = x[i]
        Y[index] = y[j]
        Z[index] = ackley_function(np.array([x[i], y[j]]))


print("X=", X)
print("Y=", Y)
print("Z=", Z)
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_trisurf(X, Y, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Ackley's Function Landscape")

# Show the plot
plt.show()