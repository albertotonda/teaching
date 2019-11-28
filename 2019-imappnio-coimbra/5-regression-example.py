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
#from sklearn.linear_model import LinearRegression
#model = LinearRegression()

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
#model = RandomForestRegressor(n_estimators=10)

# Bagging and Boosting are just other ways of creating ensembles
#from sklearn.ensemble import AdaBoostRegressor
#model = AdaBoostRegressor(n_estimators=10)

#from sklearn.ensemble import BaggingRegressor
#model = BaggingRegressor(n_estimators=10)

# and here we have a "deep" neural network with 3 layers...
#from keraswrappers import ANNRegressor
#model = ANNRegressor(epochs=100, batch_size=32, layers=[128, 32, 8])

# R2 is just a way to score our results: close to 1.0 is great, close to 0.5 is kinda bad, close to 0.0 is awful and < 0.0 is TERRIBLE

# here is a visual representation of what is going on, using a random split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model.fit(X_train, y_train)
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

from sklearn.metrics import r2_score
print("R2 score on training for this split: %.4f" % r2_score(y_train, y_train_predicted))
print("R2 score on test for this split: %.4f" % r2_score(y_test, y_test_predicted))

import matplotlib.pyplot as plt
plt.scatter(y_train, y_train_predicted, color='red', label='Training')
plt.scatter(y_test, y_test_predicted, color='green', label='Test')
x_min = min(min(y_test), min(y_train))
x_max = max(max(y_test), max(y_train))
y_min = min(min(y_test_predicted), min(y_train_predicted))
y_max = max(max(y_test_predicted), max(y_train_predicted))
plt.plot([-2,2], [-2,2], '--', color='black')
plt.legend(loc='best')
plt.show()

