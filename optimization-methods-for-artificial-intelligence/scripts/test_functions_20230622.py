# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:04:45 2023

@author: Alberto
"""

from scipy.optimize import minimize

import matplotlib.pyplot as plt # library for scientific plotting
from mpl_toolkits.mplot3d import Axes3D # used for 3d plots
import numpy as np # library with utility functions for numerical computation
import seaborn as sns # library for scientific plotting, but prettier

sns.set_style() # set the seaborn style for plots

# function that creates a figure with the two-dimensional search space of a function
def visualize_2d(function, boundaries, sampling_step=0.01, global_optimum=None) :

  # uniformely sample the only variable of the function
  x = np.arange(min(boundaries), max(boundaries), sampling_step).reshape(-1, 1)
  y = function(x, global_optimum=global_optimum)

  # create a figure and a sub-plot (there is only one sub-plot)
  fig = plt.figure()
  ax = fig.add_subplot(111)

  # plot x, y points, as a continuous line
  ax.plot(x, y)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_title("Plot for function %s, 1D" % function.__name__)

  # return a reference to the figure and the sub-figure
  return fig, ax

# function that visualizes the search space with a figure in three dimensions
def visualize_3d(function, boundaries, sampling_step=0.1, global_optimum=None) :

  # Generate input values for x, y, and calculate z; this part is a bit
  # messy to understand, as it uses a numpy function
  x = np.arange(min(boundaries), max(boundaries), sampling_step)
  y = np.arange(min(boundaries), max(boundaries), sampling_step)
  X, Y = np.meshgrid(x, y)
  Z = function(np.array([X, Y]).T, global_optimum=global_optimum)

  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot the surface, using a colormap that employs different colors for different
  # values of the objective function
  cplot = ax.plot_surface(X, Y, Z, cmap='viridis')

  # also set a colorbar for the plot
  fig.colorbar(cplot, ax=ax)

  # Set labels and title
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('f(x1,x2)')
  ax.set_title('Function %s, 2D' % function.__name__)

  #figure = plt.figure()
  #ax = figure.add_subplot(111, projection='3d')
  #ax.plot_surface(X, Y, Z, cmap='viridis', shade='false')

  return fig, ax

# function that visualizes the fitness landscape of a function, seen "from above"
def visualize_3d_from_above(function, boundaries, sampling_step=0.1, global_optimum=None) :

  # sample search space uniformely
  x = np.arange(min(boundaries), max(boundaries), sampling_step)
  y = np.arange(min(boundaries), max(boundaries), sampling_step)
  X, Y = np.meshgrid(x, y)
  Z = function(np.array([X, Y]).T, global_optimum=global_optimum)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  # Create a contour plot
  contour = ax.contour(X, Y, Z, levels=100, cmap='viridis')

  # also set a colorbar for the plot
  fig.colorbar(contour, ax=ax)

  ax.set_title("Function %s, 2D, seen from above" % function.__name__)
  ax.set_xlabel("x1")
  ax.set_xlabel("x2")

  return fig, ax


def function_sphere(x, global_optimum=None):
    if global_optimum is not None :
        return np.sum( np.square(x - global_optimum), axis=-1)
    else :
        return np.sum(x**2, axis=-1)
    
def function_ackley(x, global_optimum=None):
    n = x.shape[-1]  # Number of dimensions

    if global_optimum is not None :
        sum_sq_term = np.sum((x-global_optimum)**2, axis=-1)
        cos_term = np.cos(2*np.pi*(x-global_optimum)).sum(axis=-1)
    else :
        sum_sq_term = np.sum(x**2, axis=-1)
        cos_term = np.cos(2*np.pi*x).sum(axis=-1)

    return -20 * np.exp(-0.2 * np.sqrt(sum_sq_term / n)) - np.exp(cos_term / n) + 20 + np.exp(1)


def function_rastrigin(x, global_optimum=None):
    A = 10
    n = x.shape[-1]  # Number of elements in the last dimension

    # Calculate the sum along the last dimension
    if global_optimum is not None :
        sum_term = np.sum((x-global_optimum)**2 - A * np.cos(2 * np.pi * (x-global_optimum)), axis=-1)
    else :
        sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

    # Calculate the final result
    result = A * n + sum_term

    return result

def function_stybinski_tang(x, global_optimum=None):
    result = np.sum(x**4 - 16*x**2 + 5*x, axis=-1)

    return result/2.0


if __name__ == "__main__" :
    
    # let's create a callback function that will record all points explored
    initial_point = [-0.5,0.5]
    points_explored = [np.array(initial_point)]
    def callback(xk, points_explored=points_explored) :
        points_explored.append(xk)
        return
    
    result = minimize(function_stybinski_tang, initial_point, bounds=[[-5,5], [-5,5]], callback=callback)
    print("Found the following minimum:", result.x)
    print("Points explored:", points_explored)
    
    figure, ax = visualize_3d_from_above(function_stybinski_tang, [-5,5])
    ax.plot([x[0] for x in points_explored], [x[1] for x in points_explored], color='red')
    plt.show()