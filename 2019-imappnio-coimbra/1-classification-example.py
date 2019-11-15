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

# TODO REMOVE THIS LINE
df = df.dropna()

# now, we select the data of the target; the values are going to be 0 (did not survive) or 1 (survived)
target_feature = 'Survived'
y = df[target_feature].values
print("Target:", y)

# and we now select the values for the features that we will use to predict the target
predictive_features = ['Age', 'Pclass', 'IsAlone', 'Deck', 'Embarked_Q', 'Embarked_S', 'Sex_male', 'Title_Miss', 'Title_Mr', 'Title_Mrs'] # 'AgeGroup_1.0', 'AgeGroup_2.0', 'AgeGroup_3.0', 'AgeGroup_4.0']
X = df[predictive_features].values
print("Predictive features:", X)

# and now, let's MACHINE LEARN THE S**T OUT OF THIS DATASET! uncomment the two lines corresponding to your favorite classification algorithm (or a random one) and LET'S GO 

# Logistic Regression, old but reliable
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Decision Tree, highly interpretable but probably kinda sucky
#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier(max_depth=2)

# Random Forest, fast to train and often quite effective (albeit not always the best)
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=10)

# Support Vector Machines, oh man, this one is doing something extremely smart with data transformation!
#from sklearn.svm import SVC
#model = SVC()

# Adaptive Gradient Boosting is slow but effective
#from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=10)

# 'fit' trains the model based on the data
model.fit(X, y)
# 'predict' performs predictions
y_predicted = model.predict(X)
# and finally, we can compute the classification accuracy, in (0.0, 1.0)
from sklearn.metrics import accuracy_score
score = accuracy_score(y, y_predicted)
print("The classification accuracy score of classifier %s is %.4f" % (model.__class__.__name__, score))
