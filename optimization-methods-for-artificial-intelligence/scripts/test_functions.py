import matplotlib.pyplot as plt # library for scientific plotting
from mpl_toolkits.mplot3d import Axes3D # used for 3d plots
import numpy as np # library with utility functions for numerical computation
import seaborn as sns # library for scientific plotting, but prettier

sns.set_style() # set the seaborn style

# # function that samples the search space of a function
# def sample_function(function, n_dimensions, boundaries, sampling_step=0.01) :
  
#   # sample the search space uniformely in all dimensions
#   x_d = np.arange(min(boundaries), max(boundaries), sampling_step)

#   # generate all points to be sampled
#   y = np.zeros(x_d.shape[0]**n_dimensions)

#   if n_dimensions == 1 :
#     y = np.vectorize(function)(x_d)
#     return x_d, y

#   elif n_dimensions == 2 :
#     x1, x2 = np.meshgrid(x_d, x_d)
#     y = np.vectorize(function)(np.array([x1, x2]))
#     return x1, x2, y

#   return # it should never get here

# # function that creates a figure with the two-dimensional search space of a function
# def visualize_2d(function, boundaries, sampling_step=0.01) :

#   x, y = sample_function(function, 1, boundaries, sampling_step)

#   figure = plt.figure()
#   ax = figure.add_subplot(111)
#   ax.plot(x, y)
#   ax.set_xlabel("x")
#   ax.set_ylabel("y")

#   return figure

# # function that visualizes the search space with a figure in three dimensions
# def visualize_3d(function, boundaries, sampling_step=0.1) :

#   x1, x2, y = sample_function(function, 2, boundaries, sampling_step)

#   figure = plt.figure()
#   ax = figure.add_subplot(111, projection='3d')
#   ax.plot_trisurf(x1, x2, y, cmap='viridis', shade='false')

#   return


# def function_sphere(x) :
#   if np.isscalar(x) :
#     x = np.array([x])
#   d = x.shape[0]
  
#   f = 0.0
#   for i in range(0, d) :
#     f += x[i] ** 2

#   return f

# boundaries_sphere = [-1, 1]
# figure = visualize_2d(function_sphere, boundaries_sphere)
# figure_3d = visualize_3d(function_sphere, boundaries_sphere)

def visualize_3d(function, boundaries, sampling_step=0.1) :

  #x1, x2, y = sample_function(function, 2, boundaries, sampling_step)
  #x = np.arange(min(boundaries), max(boundaries), sampling_step)
  #y = np.arange(min(boundaries), max(boundaries), sampling_step)
  #meshgrid = np.meshgrid(x, y)
  #X, Y = meshgrid
  #Z = function(np.array(meshgrid).reshape(1,-1))

  # Generate input values for x, y, and calculate z
  x = np.arange(min(boundaries), max(boundaries), sampling_step)
  y = np.arange(min(boundaries), max(boundaries), sampling_step)
  X, Y = np.meshgrid(x, y)
  argument = np.array([X, Y]).T
  Z = function(argument)

  print("x shape:", x.shape)
  print("X shape:", X.shape)
  print("Y shape:", Y.shape)
  print("argument shape:", argument.shape)
  print("Z shape:", Z.shape)

  # Create a 3D plot
  figure = plt.figure()
  ax = figure.add_subplot(111, projection='3d')

  # Plot the surface
  ax.plot_surface(X, Y, Z, cmap='viridis')

  # Set labels and title
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('f(x1,x2)')
  ax.set_title('Sphere Function Landscape')

  #figure = plt.figure()
  #ax = figure.add_subplot(111, projection='3d')
  #ax.plot_surface(X, Y, Z, cmap='viridis', shade='false')

  return figure, ax

def function_ackley(x, global_optimum=None):
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



figure, ax = visualize_3d(function_ackley, [-5,5])
plt.show()