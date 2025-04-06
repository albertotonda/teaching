# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def dirac_delta(x, y, x0, y0):
#     return 1 if x == x0 and y == y0 else 0

# def multimodal_function(x, y):
#     return np.exp(-(x - 2)**2 - (y - 2)**2) + 0.8 * np.exp(-(x + 2)**2 - (y + 2)**2)

# # Define the range and step size for the x and y axes
# x_start, x_end, x_step = -5, 5, 0.1
# y_start, y_end, y_step = -5, 5, 0.1

# # Create a meshgrid for the x and y axes
# x_range = np.arange(x_start, x_end, x_step)
# y_range = np.arange(y_start, y_end, y_step)
# X, Y = np.meshgrid(x_range, y_range)

# # Compute the Dirac's Delta values for each point in the meshgrid
# delta_values = np.vectorize(dirac_delta)(X, Y, 0, 0)

# # Compute the multimodal function values for each point in the meshgrid
# function_values = multimodal_function(X, Y)

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the Dirac's Delta as an arrow pointing up
# arrow_scale = 0.1
# arrow_indices = np.where(delta_values == 1)
# for x, y in zip(X[arrow_indices], Y[arrow_indices]):
#     ax.quiver(x, y, 0, 0, 0, arrow_scale, color='red', arrow_length_ratio=0)

# # Plot the multimodal function
# #ax.plot_surface(X, Y, function_values, cmap='viridis', alpha=0.8, label='Multimodal Function')

# # Set plot labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title("Dirac's Delta and Multimodal Function in 3D")

# # Show the 3D plot
# plt.savefig("smoothen.png", dpi=300)
# plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the coordinates of the Dirac's Delta
# x = 0
# y = 0
# z = 0

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the Dirac's Delta as an arrow pointing up
# arrow_scale = 0.5
# ax.quiver(x, y, z, 0, 0, arrow_scale, color='red', arrow_length_ratio=0)

# # Set plot labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title("Dirac's Delta in 3D")

# # Set plot limits
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([0, 1])

# # Show the 3D plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dirac_delta(x, y, x0, y0):
    return 1 if x == x0 and y == y0 else 0

def multimodal_function(x, y):
    return np.exp(-(x - 2)**2 - (y - 2)**2) + 0.8 * np.exp(-(x + 2)**2 - (y + 2)**2)

# Define the coordinates of the Dirac's Delta maximum
x_delta = 0
y_delta = 0
z_delta = 1

# Define the range for the x and y axes
x_start, x_end, x_step = -5, 5, 0.1
y_start, y_end, y_step = -5, 5, 0.1

# Create a meshgrid for the x and y axes
x_range = np.arange(x_start, x_end, x_step)
y_range = np.arange(y_start, y_end, y_step)
X, Y = np.meshgrid(x_range, y_range)

# Compute the Dirac's Delta values for each point in the meshgrid
delta_values = np.vectorize(dirac_delta)(X, Y, x_delta, y_delta)

# Compute the multimodal function values for each point in the meshgrid
function_values = multimodal_function(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Dirac's Delta as an arrow pointing up
arrow_scale = 0.5
ax.quiver(x_delta, y_delta, z_delta, 0, 0, arrow_scale, color='red', arrow_length_ratio=0)

# Plot the multimodal function
ax.plot_surface(X, Y, function_values, cmap='viridis', alpha=0.5)

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Dirac's Delta and Multimodal Function in 3D")

# Set plot limits
ax.set_xlim([x_start, x_end])
ax.set_ylim([y_start, y_end])
ax.set_zlim([0, np.max(function_values)])

# Show the 3D plot
plt.savefig("smoothen.png", dpi=300)
plt.show()

