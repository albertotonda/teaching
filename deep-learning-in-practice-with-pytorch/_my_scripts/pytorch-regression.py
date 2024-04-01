# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:08:53 2024

@author: Alberto
"""

import torch

import matplotlib.pyplot as plt
import numpy as np
import openml
import sklearn

def load_and_preprocess_dataset(dataset_id=189) :
    
    dataset = openml.datasets.get_dataset(dataset_id)

    df, *_ = dataset.get_data()
    print(df)
    
    # as you noticed, some of the columns contain strings instead of numbers; we call these
    # "categorical" columns or features. We need to change that, as most ML algorithms
    # only process numbers. Don't worry too much about this part, it's just replacing strings
    # with numbers
    categorical_columns = df.select_dtypes(include=['category', 'object', 'string']).columns
    for c in categorical_columns :
      df[c].replace({category : index for index, category in enumerate(df[c].astype('category').cat.categories)}, inplace=True)
    
    # also, remove all rows that contain invalid numerical values (for example, missing values)
    df.dropna(inplace=True)
    
    # name of the target column
    target_feature = dataset.default_target_attribute
    other_features = [c for c in df.columns if c != target_feature]
    
    # get just the data without the column headers, as numerical matrices
    X = df[other_features].values
    y = df[target_feature].values
    
    from sklearn.model_selection import train_test_split
    # we use 10% of the data for the test, the rest for training and validation
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    # we use 20% of the remaining data for validation, the rest for the training
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training, test_size=0.2, shuffle=True, random_state=42)

    # StandardScaler is an object that is able to learn and apply a normalization
    # that reduces the values of a feature to zero mean and unitary variance
    # so that most of the data values of a feature will fall into the interval (-1,1)
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler() # we need a separate instance of the StandardScaler object for X and y
    scaler_y = StandardScaler()
    
    scaler_X.fit(X_train)
    scaler_y.fit(y_train.reshape(-1,1))
    
    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # don't worry too much about all this reshaping going on here, these function like
    # to have their inputs in a particular way, but the functions later like another
    # type of input, so we are forced to reshape and reshape again
    y_train_scaled = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1,)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1,1)).reshape(-1,)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1,)
    
    # convert all the arrays to pytorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float)
    
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float)
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

# we create a new class, that inherits from torch.nn.Module, the basic pytorch module
class TwoLayerNeuralNetwork(torch.nn.Module) :

  # first, the builder __init__ that is called every time the class is instantiated
  # note that we added an additional argument, input_features_size, that we can modify
  # to adapt this to problems with a different number of features
  def __init__(self, input_features_size=8):
    super(TwoLayerNeuralNetwork,self).__init__() # call the __init__ of the parent class Module
    self.linear_1 = torch.nn.Linear(input_features_size, 5) # linear layer, input_features_size inputs and 5 outputs
    self.activation_function_1 = torch.nn.Sigmoid() # activation function
    self.linear_2 = torch.nn.Linear(5, 1) # another linear layer, 5 inputs, 1 output (that will be intepreted as the prediction)

  # the method 'forward' describes what happens during a forward pass
  def forward(self, x) :
    z_1 = self.linear_1(x) # pass inputs through first linear module
    z_2 = self.activation_function_1(z_1) # pass output of linear layer through activation function
    y_hat = self.linear_2(z_2) # pass output of activation function through last linear TwoLayerNeuralNetwork

    # return the tensor in output of the last module as a prediction
    return y_hat

if __name__ == "__main__" :
    
    # this should be an easy example of pytorch code
    print("pytorch version:",torch.__version__)
    print("Is GPU computation available?", torch.cuda.is_available())

    # this is only relevant for people with a MacBook M1 and pytorch >= 2.0
    print("(Mac M1 only) Is GPU computation available?", torch.backends.mps.is_available())

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_dataset()
    
    torch.manual_seed(42)
    two_layer_neural_network = TwoLayerNeuralNetwork()
    
    learning_rate = 1e-1

    # the number of 'epochs' is the number of iterations of gradient descent after
    # which the algorithm will stop and return the best solution found so far
    max_epochs = 2000
    
    # we select the type of loss we are going to use; in this case, it's going to be
    # Mean Squared Error (MSE), appropriate for most regression tasks
    mse_loss = torch.nn.MSELoss()
    
    # and now we can start the iterative optimization loop; we instantiate the optimizer
    # Stochastic Gradient Descent (SGD) to optimize the parameters of the network
    optimizer = torch.optim.SGD(params=two_layer_neural_network.parameters(), lr=learning_rate)
    
    # some data structures here to store the training and validation loss
    train_losses = np.zeros((max_epochs,))
    val_losses = np.zeros((max_epochs,))
    
    # and now we start the optimization process!
    print("Starting optimization...")
    for epoch in range(0, max_epochs) :
      # get the tensor containing the network predictions for the training set,
      # using the current parameters (initially, they will all be random)
      y_train_pred = two_layer_neural_network(X_train)
    
      # compute loss
      loss_train = mse_loss(y_train_pred, y_train.view(-1,1))
      train_losses[epoch] = loss_train.item() # .item() here access the value stored inside the tensor that mse_loss returns
    
      # now, here we need to also compute the loss on the validation set; we use the
      # context torch.no_grad() to skip all computations on the tensor during the
      # forward pass, we do not need to store that information for the validation set
      # TODO: you have to code this
      with torch.no_grad() :
        y_val_pred = two_layer_neural_network(X_val)
        loss_val = mse_loss(y_val_pred, y_val.view(-1,1))
        val_losses[epoch] = loss_val.item()
    
      if epoch == 0 or epoch % 1000 == 0 :
        print("Epoch %d: training loss=%.4e, validation loss=%.4e" % (epoch, loss_train, loss_val)) # this printout is run only each 100 epochs
    
      # set the cumulated gradients back to zero (to avoid cumulating from one epoch to the next)
      optimizer.zero_grad()
      # perform the backward operation to retropropagate the error and get the gradient
      loss_train.backward()
      # perform one step of the gradient descent, modifying the network parameters
      optimizer.step()

    from sklearn.metrics import r2_score
    
    with torch.no_grad() :
        y_train_pred = two_layer_neural_network(X_train)
        r2_train = r2_score(y_train_pred.numpy(), y_train.numpy())
        
        y_val_pred = two_layer_neural_network(X_val)
        r2_val = r2_score(y_val_pred, y_val)
        
        y_test_pred = two_layer_neural_network(X_test)
        r2_test = r2_score(y_test_pred.numpy(), y_test.numpy())
        
        print("R2 on training: %.4f" % r2_train)
        print("R2 on validation: %.4f" % r2_val)
        print("R2 on test: %.4f" % r2_test)
        
    # this is used for the x-axis of the plot
    x_epochs = [i for i in range(0, max_epochs)]
    
    fig, ax = plt.subplots()
    ax.plot(x_epochs, train_losses, color='orange', label="Training loss")
    ax.plot(x_epochs, val_losses, color='green', label="Validation loss")
    ax.legend(loc='best')
    ax.set_title("Performance on training: R2=%.4f, on test: R2=%.4f" % (r2_train, r2_test))