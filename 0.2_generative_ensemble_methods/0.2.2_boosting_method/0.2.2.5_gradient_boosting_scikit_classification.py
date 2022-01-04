""" 
- Scikit-learn also implements gradient boositng regression and classification. 
- The ensemble package under GradientBoostingRegressor and GradientBoostingClassifier, respectively. 
- The two classes store the errors at each step in the train_score_attribute of the object
 """

# Step 1: Importing the libraries and data
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

digits = load_digits()

train_size = 1500

# Split the train and test set
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

# Step 2: Create the ensemble
ensemble_size =200
learning_rate = 0.1
ensemble = GradientBoostingClassifier(n_estimators=ensemble_size, learning_rate=learning_rate)

# Step 3: Evaluate the ensemble
ensemble.fit(train_x, train_y)
ensemble_predictions = ensemble.predict(test_x)
ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Step 4: print accuracy
print('Boosting: %.2f '% ensemble_acc)
# Boosting: 0.88 

# The accuracy achieved with the specific ensemble size is 88%.