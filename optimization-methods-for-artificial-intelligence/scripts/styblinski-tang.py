# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:57:51 2023

Script to plot the Styblinski-Tang function

@author: alberto.tonda@inrae.fr
"""
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

def styblinski_tang(x) :
    """
    Evaluate the Styblinski-Tang function in one point.
    """
    d = len(x)
    y = 0
    for i in range(0, d) :
        y += x[i]**4 - 16 * x[i]**2 + 5 * x[i]
        
    y = 1/2 * y
    
    return y

if __name__ == "__main__" :
    
    print("Creating meshgrid...")
    points = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(points, points)
    
    print("Computing function values...")
    y = np.array([styblinski_tang([x1[i], x2[i]]) for i in range(0, len(x1))])
    
    print("Plotting figure...")
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    
    #ax.contour3D(x1, x2, y, 50, cmap='RdYlGn')
    ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='RdYlGn', edgecolor='none')

    # let's first set the view and save three images
    def animate(i):
        ax.view_init(elev=49., azim=i)
        return fig,

    def init():
        ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap='RdYlGn', edgecolor='none')
        return fig,

    views = [30, 60, 90, 180]
    for v in views :
        animate(v)
        plt.savefig("styblinski-tang-%d.png" % v, dpi=300)

    from matplotlib import animation

    print("Creating animation...")
    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
    # Save
    anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
    # def f(x, y):
    #     return np.sin(np.sqrt(x ** 2 + y ** 2))

    # x = np.linspace(-6, 6, 30)
    # y = np.linspace(-6, 6, 30)
    
    # X, Y = np.meshgrid(x, y)
    # Z = f(X, Y)
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    
        
