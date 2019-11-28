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

# also, let's normalize the features
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# and now, let's MACHINE LEARN THE S**T OUT OF THIS DATASET! uncomment the two lines corresponding to your favorite classification algorithm (or a random one) and LET'S GO 

# Logistic Regression, old but reliable
#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()

# Support Vector Machines, oh man, this one is doing something extremely smart with data transformation!
#from sklearn.svm import SVC
#model = SVC()

# Decision Tree, highly interpretable but probably kinda sucky
#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier(max_depth=2)

# Random Forest, fast to train and often quite effective (albeit not always the best)
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=10)

# Adaptive Gradient Boosting is slow but effective
#from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=10)

# Bagging is more or less as slow and effective as AdaBoost
#from sklearn.ensemble import BaggingClassifier
#model = BaggingClassifier(n_estimators=10)

# Artificial (Deep) Neural Network!
#from keraswrappers import ANNClassifier
#model = ANNClassifier(layers=[128], batch_size=32, epochs=100, learning_rate=1e-5)

# DO NOT DECOMMENT THIS ONE, used for later
#from tpot import TPOTClassifier
#model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

# what if we want to know WHY is the classifier taking these decisions?
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

model.fit(X, y)
features_scored_by_importance = model.feature_importances_
features_sorted_by_importance = [predictive_features[index] for score, index in sorted(zip(features_scored_by_importance, range(0, len(features_scored_by_importance))), reverse=True)]
print(features_sorted_by_importance)

# we can actually visualize a decisione tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
model = DecisionTreeClassifier(max_depth=2)

import matplotlib.pyplot as plt
model.fit(X, y)
plot_tree(model, feature_names=predictive_features, class_names=['Dead', 'Alive'])
plt.show()
