"""
- Apart from conventional Random Forest, scikit-learn also implements Extra Trees. 
- The classification implementaion lies in the ExtraTreesClassifier in the sklearn.ensemble package
 """

# Step 1: Importing libraries and dataset
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
import numpy as np

digits  =  load_digits()

train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

""" 
- Following this we will create the ensemble by setting the n_estimators and n_jobs parameters. 
- These parameters dictate the number of trees that will be generated and the number of parallel jobs that 
                will be run. 
                We train the ensemble using the fit function and evaluate it on the test set by measuring its 
                achieved accuracy """

# Step 2: create the ensemble
ensemble_size = 500
ensemble = ExtraTreesClassifier(n_estimators=ensemble_size, n_jobs=4)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
# ExtraTreesClassifier(n_estimators=500, n_jobs=4)

# Step 4: Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)
ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Step 5: Print the accuracy
print('Random Forest: %.2f' % ensemble_acc)
""" 
Random Forest: 0.94

- The classifier achieve 93% which is higher than previous best performing method (XGBoost)
- The extra Tree classifier achieved 94% which is even better than the Random Forest classifier """