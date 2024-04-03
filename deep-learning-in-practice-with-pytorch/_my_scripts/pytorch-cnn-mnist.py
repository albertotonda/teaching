# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:04:24 2024

@author: Alberto
"""

import torch
import torchvision # a part of the pytorch project, exclusively dedicated to image and video analysis

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" :
    # with torchvision.transforms.Compose, we can create a sequence of transformations that will be applied to the data,
    # in order; in this case, we first turn everything into Tensors, then we normalize between 0.0 and 1.0
    mnist_transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.,), (1.0,))])
    
    # instantiate the MNIST dataset, and apply the sequence of transformations we created above
    # the download=True flag will download the data if it is not already present; train=True or =False downloads a training set or a test set, respectively
    # root is just the folder where the data should be downloaded
    train_set = torchvision.datasets.MNIST(root=".", train=True, transform=mnist_transformations, download=True)
    test_set = torchvision.datasets.MNIST(root=".", train=False, transform=mnist_transformations, download=True)
    
    mnist_test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
    
    for X_test, y_test in mnist_test_loader :
        print(X_test.shape)
        print(y_test.shape)