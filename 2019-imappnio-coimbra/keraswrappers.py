# This is a simple wrapper for a keras-based artificial neural network. I programmed it from scratch because I am experiencing issues
# TODO create another wrapper for a keras-based classifier
# with keras' native class KerasRegressor. by Alberto Tonda, 2018 <alberto.tonda@gmail.com>

import numpy as np
import sys

# and here is the keras-related stuff
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# scikit-learn stuff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

# artificial neural network performing regression tasks
class ANNRegressor :
    
    # attributes
    model = None
    epochs = 1000
    batch_size = 32
    layers = [32, 8]

    # size of layers is a list of integers
    # TODO if possible, overload and/or specify default values for epochs, batch_size, layers
    def __init__(self, epochs, batch_size, layers) :
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
    
    def fit(self, X, y) :
        #print("Shape of X:", X.shape)
        #print("Shape of y:", y.shape)
        self.model = Sequential()
        self.model.add(Dense(self.layers[0], input_dim=X.shape[1], activation='relu'))
        for i in range(1, len(self.layers)) : self.model.add(Dense(self.layers[i], activation='relu'))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)
    
    def predict(self, X) :
        if self.model != None :
            return self.model.predict(X)
        else :
            sys.stderr.write("Error: model has not been trained. Call 'fit' method before 'predict'.")
            return None

    # this function here is to make it compliant with scikit-learn
    def get_params(self, **params):

        new_dict = dict()
        new_dict["epochs"] = self.epochs
        new_dict["batch_size"] = self.batch_size
        new_dict["layers"] = self.layers

        return new_dict

# artificial neural network performing classification tasks: basically the same as above, with a softmax in the last layer
# also, the "score" method is implemented, returning accuracy
class ANNClassifier :
    
    # attributes
    model = None
    epochs = 1000
    batch_size = 32
    layers = [32, 8]
    learning_rate = 1e-6
    sample_weights = None

    # size of layers is a list of integers
    # TODO if possible, overload and/or specify default values for epochs, batch_size, layers
    def __init__(self, epochs=1000, batch_size=32, layers=[32,8], learning_rate=1e-6) :
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.learning_rate = learning_rate

    def __str__(self) :
        temp_string = "ANNClassifier"
        for l in self.layers : temp_string += "_%d" % l
        return temp_string
    
    def fit(self, X, y, sample_weight=None) :
        #print("Shape of X:", X.shape)
        #print("Shape of y:", y.shape)
        
        # first, we need to find the number of different classes, as the last layer needs
        # to have a corresponding number of nodes
        numberOfClasses = len(np.unique(y))
        
        self.model = Sequential()
        self.model.add(Dense(self.layers[0], input_dim=X.shape[1], activation='relu'))
        for i in range(1, len(self.layers)) : self.model.add(Dense(self.layers[i], activation='relu'))
        self.model.add(Dense(numberOfClasses, activation='softmax'))

        optimizer = optimizers.RMSprop(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        oneHotLabels = to_categorical(y, num_classes=numberOfClasses)
        self.model.fit(X, oneHotLabels, epochs=self.epochs, batch_size=self.batch_size, sample_weight=sample_weight)
    
    def predict(self, X) :
        if self.model != None :
            y_pred = [ np.argmax(y_p) for y_p in self.model.predict(X) ]
            return y_pred
        else :
            sys.stderr.write("Error: model has not been trained. Call 'fit' method before 'predict'.")
            return None
    
    def score(self, X, y) :
        # since the network outputs a prediction vector of N_classes values, we need to map
        # the predictions to a class, finding the index of the element with the
        # highest value in each prediction vector returned by the network 
        y_pred_temp = self.predict(X) 
        y_pred = [ np.argmax(y_p) for y_p in y_pred_temp ]
        
        score = 0.0
        for i in range(0, len(y)) :
            if y_pred[i] == y[i] :
                    score += 1.0
        score /= len(y)
        
        return score

    # this function here is to make it compliant with scikit-learn
    def get_params(self, **params):

        new_dict = dict()
        new_dict["epochs"] = self.epochs
        new_dict["batch_size"] = self.batch_size
        new_dict["layers"] = self.layers
        new_dict["learning_rate"] = self.learning_rate

        return new_dict

# this "main" function is just here to test the code, it's not meant to be really used
def main() :
    
    # test classification
    if True :
            X, y, variablesX, variablesX = commonLibrary.loadAlejandroNewDataset()
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=10, shuffle=True) 
            for train_index, test_index in skf.split(X, y) :
                    
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    

                    scalerX = StandardScaler()
                    X_train = scalerX.fit_transform(X_train)
                    X_test = scalerX.transform(X_test)
                    
                    classifier = ANNClassifier(epochs=50, batch_size=128, layers=[128,64])
                    classifier.fit(X_train, y_train)
                    
                    scoreTraining = classifier.score(X_train, y_train)
                    scoreTest = classifier.score(X_test, y_test)
                    
                    print("Fold, training: %.4f, test: %.4f" % (scoreTraining, scoreTest))

    # test regression
    if False :
            X, y, variablesX, variablesY = commonLibrary.loadEcosystemServices()
            print("Variables X (" + str(len(variablesX)) + "):", variablesX)

            scalerX = StandardScaler()
            scalerY = StandardScaler()

            X_train = scalerX.fit_transform( X ) 
            y_train = scalerY.fit_transform( y[:,0].reshape(-1,1) )
            
            regressor = ANNRegressor(epochs=1000, batch_size=32)
            regressor.fit(X_train, y_train)
            
            y_predict = regressor.predict(X_train)
            
            print("R^2 score (training):", r2_score(y_train, y_predict))
            print("MSE score (training):", mean_squared_error(y_train, y_predict))
            print("EV score (training):", explained_variance_score(y_train, y_predict))

    return

if __name__ == "__main__" :
    sys.exit( main() )
