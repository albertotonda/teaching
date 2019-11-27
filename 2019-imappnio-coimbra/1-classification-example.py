# Example of classification, with several subsequent steps, to be later cut/pasted to a Colab sheet
# by Alberto Tonda, 2019 <alberto.tonda@gmail.com>

# the Titanic dataset can be found on multiple sites (for example, OpenML); but here, we will load a pre-processed version
# that already dealt with all the boring stuff, such as filling in empty cells, turning categories into numbers, etc.
# we use the Pandas module to load a CSV directly from a github repository into a Pandas' 'dataframe' object
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/albertotonda/teaching/master/2019-imappnio-coimbra/titanic_dataset_preprocessed.csv', sep=',')

# Pandas' dataframes are very convenient for manipulating datasets
print(df.head())
print("The dataset contains %d samples and %d features." % (df.shape[0], df.shape[1]))

# we can for example count the number of passengers that survived (feature 'Survived' == 1)
print("Of the %d passengers in the dataset, %d survived." % (df.shape[0], df[ df['Survived'] == 1].shape[0]))

# now, we select the data of the target; the values are going to be 0 (did not survive) or 1 (survived)
target_feature = 'Survived'
y = df[target_feature].values
print("Target:", y)

# and we now select the values for the features that we will use to predict the target
predictive_features = ['Age', 'Pclass', 'IsAlone', 'Deck', 'Embarked_Q', 'Embarked_S', 'Sex_male', 'Title_Miss', 'Title_Mr', 'Title_Mrs'] # 'AgeGroup_1.0', 'AgeGroup_2.0', 'AgeGroup_3.0', 'AgeGroup_4.0']
X = df[predictive_features].values
print("Predictive features:", X)

# we split the data into training and test (we will use test data later)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# and now, let's MACHINE LEARN THE S**T OUT OF THIS DATASET! uncomment the two lines corresponding to your favorite classification algorithm (or a random one) and LET'S GO 

# Logistic Regression, old but reliable
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()

# Decision Tree, highly interpretable but probably kinda sucky
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)

# Random Forest, fast to train and often quite effective (albeit not always the best)
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=10)

# Support Vector Machines, oh man, this one is doing something extremely smart with data transformation!
#from sklearn.svm import SVC
#model = SVC()

# Adaptive Gradient Boosting is slow but effective
#from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=10)

# Bagging is more or less as slow and effective as AdaBoost
#from sklearn.ensemble import BaggingClassifier
#model = BaggingClassifier(n_estimators=10)

# 'fit' trains the model based on the data
model.fit(X_train, y_train)
# 'predict' performs predictions
y_train_predicted = model.predict(X_train)
# and finally, we can compute the classification accuracy, in (0.0, 1.0)
from sklearn.metrics import accuracy_score
score_train = accuracy_score(y_train, y_train_predicted)
print("The classification accuracy score of classifier %s on the training data is %.4f" % (model.__class__.__name__, score_train))

# now, let's see what happens with the classification accuracy on test (unseen data)
y_test_predicted = model.predict(X_test)
score_test = accuracy_score(y_test, y_test_predicted)
print("The classification accuracy score of classifier %s on the test data is %.4f" % (model.__class__.__name__, score_test))

# to obtain a more reliable performance, a classifier can be run multiple times, using random training/test splits, and we can then take the average
n_folds = 10
print("Now performing a cross-validation with %d folds (might take a while)..." % n_folds)
from sklearn.model_selection import cross_validate
cross_validation_results = cross_validate(model, X, y, scoring='accuracy', cv=n_folds)

# numpy is a module with utility functions for quickly computing functions
import numpy as np
print("Average classification accuracy score (on test) for model %s is %.4f (+/- %.4f)" % (model.__class__.__name__, np.mean(cross_validation_results["test_score"]), np.std(cross_validation_results["test_score"])))

# what if we want to know WHY is the classifier taking these decisions?

