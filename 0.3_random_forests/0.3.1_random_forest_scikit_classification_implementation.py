""" 
- Scikit-learn implements both conventional Random Forest Trees as well as Extra Trees. 
- In this example and next few examples we will provide the basic regression and classification examples 
        with both algorithms using the scikit-learn implementation
- The Random Forests classification class is implemented in RandomForestClassifier, under the sklearn.ensemble package. 
        It has a number of parameters, such as:
            - the ensemble's size
            - the maximum tree depth
            - the number of samples required to make or split a node and 
            - many more..
- In this example we will try to classify the hand-written digits dataset, using the Random Forest Classification 
        ensemble.
 """

# Step 1: Importing libraries and dataset
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

digits  =  load_digits()

train_size = 1500

train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

""" 
- Following this we will create the ensemble by setting the n_estimators and n_jobs parameters. 
- These parameters dictate the number of trees that will be generated and the number of parallel jobs that will be run.
- We train the ensemble using the fit function and evaluate it on the test set by measuring its achieved accuracy
 """

# Step 2: create the ensemble
ensemble_size = 500
ensemble = RandomForestClassifier(n_estimators=ensemble_size, n_jobs=4)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
# RandomForestClassifier(n_estimators=500, n_jobs=4)

# Step 4: Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)
ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Step 5: Print the accuracy
print('Random Forest: %.2f' % ensemble_acc)
# Random Forest: 0.93
# The classifier achieve 93% which is higher than previous best performing method (XGBoost)