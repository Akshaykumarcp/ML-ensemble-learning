""" 
- Scikit-learn's implementation of bagging lies in the sklearn.ensemble package. 
- BaggingClassifier is the corresponding class for classification problems. 
- It has a number of interesting parameters, allowing for greater flexibility. 
- It can use any scikit-learn estimator by specifying it with base_estimator.

Furthermore, 
- n_esimators dictates the ensemble's size (and, consequently, number of bootstrap samples that will be generated), 
- while n_jobs dictates how many jobs (processes) will be used to train and predict with each base learner.

Finally if set to True, oob_score calcualtes the out-of-bag score for the base learners
"""

# Step 1: Import libraries and data
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics

digits = load_digits()

# Split the train and test set
train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

# Step 2: we need to create train and evaluate the estimator
# Create the ensemble
ensemble_size = 10
ensemble = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                             n_estimators=ensemble_size,
                             oob_score=True)

# Train the ensemble
ensemble.fit(train_x, train_y)

# Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Step 3: Print accuracy
print('Bagging: %.2f'%ensemble_acc)
Bagging: 0.86

""" 
- The final achieved accuracy is 87%. In our example we only choose an ensemble_size of 10.

Exercise:
- change the ensemble_size to be any number which is less than 10 and greater than 10 and greater than 20 To 
        find the the bagging. 
        Please compare them with mine result to see are there any differences. """