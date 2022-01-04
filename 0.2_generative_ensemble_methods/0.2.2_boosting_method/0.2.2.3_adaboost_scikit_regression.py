# Step 1: Import libraries and data
from copy import deepcopy
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics

diabetes = load_diabetes()
#Split the train and test set

train_size = 400

train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# Step 2: Create the ensemble
ensemble_size = 1000
ensemble = AdaBoostRegressor(n_estimators=ensemble_size)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
# AdaBoostRegressor(n_estimators=1000)

#Step 4: Evaluate the ensemble
predictions = ensemble.predict(test_x)

# Step 5: Print metrics
r2 = metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)
print('Gradient Boosting: ')
print('R-squared: %.2f' % r2)
print('MSE: %.2f'%mse)
""" 
Gradient Boosting: 
R-squared: 0.56
MSE: 2417.89
The ensemble generates an R-squared of 0.56 and an MSE of 2417.89. """