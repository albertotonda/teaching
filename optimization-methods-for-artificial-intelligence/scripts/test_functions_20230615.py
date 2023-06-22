# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:25:53 2023

@author: Alberto
"""
import matplotlib.pyplot as plt
import numpy as np

def ackleys_function(x, global_optimum=None):
    n = x.shape[-1]  # Number of dimensions
    
    if global_optimum is not None :
        sum_sq_term = np.sum((x-global_optimum)**2, axis=-1)
        cos_term = np.cos(2*np.pi*(x-global_optimum)).sum(axis=-1) 
    else :
        sum_sq_term = np.sum(x**2, axis=-1)
        cos_term = np.cos(2*np.pi*x).sum(axis=-1)     
    
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq_term / n)) - \
           np.exp(cos_term / n) + 20 + np.exp(1)

def sphere_function(x, global_optimum=None):
    if global_optimum is not None :
        return np.sum( np.square(x - global_optimum), axis=-1)
    else :
        return np.sum(x**2, axis=-1)

def rastrigin(x, global_optimum=None):
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

def stybinski_tang(x, global_optimum=None):
    
    if global_optimum is not None :
        true_optimum = np.full(x.shape, -2.093534)
        delta_optimum = np.abs(global_optimum - true_optimum)  
        result = np.sum( (x-delta_optimum)**4 - 16*(x-delta_optimum)**2 + 5*(x-delta_optimum), axis=-1 )
    else :
        result = np.sum(x**4 - 16*x**2 + 5*x, axis=-1)
    
    return result/2.0

plot_3d = False
plot_2d = False
plot_from_above = True
x = np.linspace(-5, 5, 100).reshape(-1, 1)    

if plot_from_above :
    X, Y = np.meshgrid(x, x)
    #Z = sphere_function(np.array([X, Y]).T, global_optimum=np.array([-2, -2]))
    #Z = ackleys_function(np.array([X, Y]).T, global_optimum=np.array([-2, -2]))
#    Z = rastrigin(np.array([X, Y]).T, global_optimum=np.array([-1, -1]))
    
    Z = stybinski_tang(np.array([X, Y]).T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.contour(X, Y, Z, levels=300, cmap='viridis')
    
    plt.show()

if plot_2d :
    y = sphere_function(x, global_optimum=np.array([0.0]))
    y = stybinski_tang(x)
    #y = ackleys_function(x)
    #y = rastrigin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()

if plot_3d :
    X, Y = np.meshgrid(x, x)
    Z = sphere_function(np.array([X, Y]).T)
    Z = rastrigin(np.array([X, Y]).T)
    Z = ackleys_function(np.array([X, Y]).T)
    Z = stybinski_tang(np.array([X, Y]).T, global_optimum=[0,0])
    
    print("Z.shape:", Z.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')