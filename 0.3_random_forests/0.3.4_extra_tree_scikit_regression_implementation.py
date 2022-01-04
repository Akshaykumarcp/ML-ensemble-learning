# We present the regression implementation of Extra Trees implemented in ExtraTreesRegressor

# Step 1: Importing libraries and dataset
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
import numpy as np

diabetes  =  load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# Step 2: create the ensemble
ensemble_size = 100
ensemble = ExtraTreesRegressor(n_estimators=ensemble_size, n_jobs=4)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
# ExtraTreesRegressor(n_jobs=4)

# Step 4: Evaluate the ensemble
predictions = ensemble.predict(test_x)

# Step 5: Print the metrics
r2= metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('Extra Trees: ')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)

""" 
Extra Trees: 
R-squared: 0.55
MSE: 2479.18

- Extra Trees outperform conventional random forests by achieving a test R-squared of 0.55 
        (0.04 better than the Random Forest) and an MSE of 2479.18(a difference of 243.49). 
        Still the ensemble seems to overfit as it perfectly predicts in-sample data. 
        By setting min_samples_leaf = 10 and the ensemble size = 1000, we are able to produce an R-suare of 0.62 
        and an MSE of 2114

Exercise:
- Change the min_samples_leaf = 10 and the ensemble size = 1000, to see of it will produce R-suare of 0.62 and 
    an MSE of 2114 ? """