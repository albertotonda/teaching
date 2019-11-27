# Example of regression, with step-by-step comments, to be later cut/pasted into a Colab sheet
# by Alberto Tonda, 2019 <alberto.tonda@gmail.com>

# 'diabetes' is a classic benchmark, coming from a real-world study, that can be found in several ML libraries/modules, such as scikit-learn
from sklearn import datasets 
X, y = datasets.load_diabetes(return_X_y=True) 
X_feature_names = ["Age", "Sex", "Body mass index", "Average blood pressure", "S1", "S2", "S3", "S4", "S5", "S6"]
y_feature_name = ["Disease progression"]
print("The dataset contains %d samples and %d features." % (X.shape[0], X.shape[1]))

# let's normalize all the features
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1,1))

# here are the 'usual suspects' in regression

# Linear regression: y = a + a_0 * X[0] + a_1 * X[1] + ...
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# 'Quadratic regression' can be implemented as a pipeline, where first new (quadratic, e.g. X[0]^2) features are created, 
# then Linear Regression is applied to these features
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
#degree = 3
#model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# And Support Vector Machines of course joins the party, basically a Linear Regression in an (artificially generated) higher dimensional space
#from sklearn.svm import SVR
#model = SVR()

# A single Decision Tree can approximate a function, by returning a fixed value on each leaf
#from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor(max_depth=2)

# Random Forest ensembles can of course be used for regression (by averaging values on the trees' leaves)
#from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor(n_estimators=300)

# Bagging and Boosting are just other ways of creating ensembles
#from sklearn.ensemble import AdaBoostRegressor
#model = AdaBoostRegressor(n_estimators=10)

#from sklearn.ensemble import BaggingRegressor
#model = BaggingRegressor(n_estimators=10)

# and here we have a "deep" neural network with 2 layers...
from keraswrappers import ANNRegressor
model = ANNRegressor(epochs=100, batch_size=32, layers=[128, 32, 8])

# now, you should know WHY we are using directly cross-validation here
# R2 is just a way to score our results: close to 1.0 is great, close to 0.5 is kinda bad, close to 0.0 is awful and < 0.0 is TERRIBLE
n_folds = 10
print("Now performing a cross-validation with %d folds (might take a while)..." % n_folds)
from sklearn.model_selection import cross_validate
cross_validation_results = cross_validate(model, X, y, scoring='r2', cv=n_folds, return_estimator=True) # return_estimator puts all trained models in the dictionary, under the key "estimator"

# let's see how we performed
import numpy as np
print("Average R2 score (on test) for model %s is %.4f (+/- %.4f)" % (model.__class__.__name__, np.mean(cross_validation_results["test_score"]), np.std(cross_validation_results["test_score"])))
