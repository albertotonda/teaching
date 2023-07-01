# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:04:45 2023

@author: Alberto
"""

from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

import matplotlib.pyplot as plt # library for scientific plotting
from mpl_toolkits.mplot3d import Axes3D # used for 3d plots
import numpy as np # library with utility functions for numerical computation
import seaborn as sns # library for scientific plotting, but prettier

import inspyred
import random

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
    if not isinstance(x, np.ndarray) :
        x = np.array(x)
    result = np.sum(x**4 - 16*x**2 + 5*x, axis=-1)

    return result/2.0


def sgd(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.001,
    mass=0.9,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of stochastic
    gradient descent with momentum.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def rmsprop(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.1,
    gamma=0.9,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of root mean
    squared prop: See Adagrad paper for details.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    avg_sq_grad = np.ones_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


def adam(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    i += 1
    return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)

if __name__ == "__main__" :
    
    # let's create a callback function that will record all points explored
    initial_point = [-0.5,0.5]
    points_explored = [np.array(initial_point)]
    def callback(xk, points_explored=points_explored) :
        points_explored.append(xk)
        return
    
    result = minimize(function_stybinski_tang, initial_point, None, bounds=[[-5,5], [-5,5]], callback=callback)
    print("Found the following minimum:", result.x)
    print("Points explored:", points_explored)
    
    figure, ax = visualize_3d_from_above(function_stybinski_tang, [-5,5])
    ax.plot([x[0] for x in points_explored], [x[1] for x in points_explored], color='red')
    plt.show()
    
    # n_dimensions = 2
    # points_explored = []
    
    # def observer(population, num_generations, num_evaluations, args) :
    #     points_explored = args["points_explored"]
    #     for individual in population :
    #         points_explored.append(individual.candidate)
    #     return
    
    # def generator(random, args) :
    #     n_dimensions = args["n_dimensions"]
    #     bounder = args["_ec"].bounder
    #     return [ random.uniform(bounder.lower_bound[i], bounder.upper_bound[i]) for i in range(0, n_dimensions) ]
    
    # from random import Random
    # prng = Random()
    # prng.seed(42)
    # ea = inspyred.ec.ES(prng)
    # ea.terminator = inspyred.ec.terminators.evaluation_termination
    # final_population = ea.evolve(generator=generator, 
    #                              evaluator=function_stybinski_tang, 
    #                              bounder=inspyred.ec.Bounder([-5] * n_dimensions, [5] * n_dimensions), 
    #                              pop_size=100, max_evaluations=100, maximize=False,
    #                              # all arguments below enter the 'args' dictionary
    #                              n_dimensions=n_dimensions,
    #                              points_explored=points_explored)
    
    # best_point = final_population[0].candidate
    
    # print("Best point found by ES:", best_point)
    # print("Points explored:", points_explored)
    
    import cma
    #print(cma.CMAOptions())
    
    def cmaes_optimization(function, starting_point, boundaries, random_seed=42, population_size=None, sigma=1e-2) :
        
        options = {'bounds' : boundaries, 'seed' : random_seed, 'popsize' : population_size}
        
        points_explored = []
        cmaes = cma.CMAEvolutionStrategy(starting_point, sigma, options)
        while not cmaes.stop():
            solutions = cmaes.ask()
            cmaes.tell(solutions, [function(x) for x in solutions])
            cmaes.disp()
            points_explored.extend(solutions)
        
        return cmaes.result[0], points_explored
    
    
    def generator(random, args) :
        boundaries = args["boundaries"]
        n_dimensions = args["n_dimensions"]
        return [random.uniform(boundaries[0], boundaries[1]) for i in range(0, n_dimensions)]
    
    def observer(population, num_generations, num_evaluations, args) :
        print("Iteration %d (%d evaluations): best point %s, function value %.4f" %
              (num_generations, num_evaluations, str(population[0].candidate), population[0].fitness))
        points_explored = args["points_explored"]
        for individual in population :
            points_explored.append(individual.candidate)
        return
    
    def evaluator(candidates, args) :
        function = args["function"]
        fitness_values = []
        for c in candidates :
            fitness_values.append(function(c))
            
        return fitness_values
    
    def evolutionary_optimization(function, starting_point, boundaries, random_seed=42, population_size=10, max_evaluations=100) :
        """
        starting_point here is not really used, it 
        """
        points_explored = []
        prng = random.Random()
        prng.seed(random_seed)
        ea = inspyred.ec.EvolutionaryComputation(prng)
        ea.observer = observer
        ea.replacer = inspyred.ec.replacers.plus_replacement
        ea.selector = inspyred.ec.selectors.tournament_selection
        ea.terminator = inspyred.ec.terminators.evaluation_termination
        ea.variators = [inspyred.ec.variators.gaussian_mutation, inspyred.ec.variators.crossovers.n_point_crossover]
        
        final_population = ea.evolve(
            pop_size=population_size, num_selected=population_size,
            generator=generator, evaluator=evaluator,
            max_evaluations=max_evaluations, maximize=False, 
            # all variables set below end up inside the 'args' dictionary
            n_dimensions=len(initial_point),
            boundaries=boundaries,
            function=function,
            points_explored=points_explored,
            )
        
        best_point = final_population[0].candidate
        
        return best_point, points_explored
    
    best_point, points_explored = cmaes_optimization(function_stybinski_tang, [1, -1], [-5,5], random_seed=None)
    figure2, ax2 = visualize_3d_from_above(function_stybinski_tang, [-5,5])
    ax2.scatter([x[0] for x in points_explored], [x[1] for x in points_explored], color='red', marker='x')
    ax2.scatter(best_point[0], best_point[1], color='red', marker='x')
    plt.show()
    