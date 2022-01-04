""" 
- XGBoost is a boosting library with parallel, GPU, and distributed execution support. 
- It has help many machine learning engineers and data scientists to win Kaggle competitions.

- Furthermore, it provides an interface that resembles scikit-learn interface. 
- Thus, someone already familar with the interface is able to quickly utilize the library. 
- Additionally, it allows for very fine control over the ensemble's creation. 
- It supports monotonic contraints (that is the predicted value should only increase or decrease, 
        relative to a specific feature), as well as feature interaction constraints (for example, if a decision tree 
        creates a node that splits by age, it should not use sex as a splitting feature for all children of 
        that specific node). 
        Finally it adds an additional regularization parameter gamma, which further reduces the overfitting 
        capabilities of the gernerated ensemble.

- We will present a simple regression example with XGBoost, using the diabetes dataset. 
        XGBoost implement regression with XGBRegressor.

- The constructor has a respectably large number of parameters which are very well documnetted in the official document,

- In our example we will use n_estimators, n_jobs, max_depth and learning_rate parameters. 
- Following scikit-learn's convention, they define the ensemble size, the number of parallel processes, 
        the tree's maximum depth and the learning rate respectively.
 """

# Step 1: Import libraries and data
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np

diabetes = load_diabetes()

train_size = 400

train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

#Step 2: Create the ensemble
ensemble_size = 200
ensemble = XGBRegressor(n_estimators=ensemble_size, n_jobs=4, max_depth=1, learning_rate=0.1, objective='reg:squarederror')

# Step 3: Evaluate the ensemble
ensemble.fit(train_x,train_y)
""" 
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=1,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=200, n_jobs=4, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None) """

predictions = ensemble.predict(test_x)

#Step 4: print the metrics
r2= metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y,predictions)
print('XGBoost: ')
print('R-squared: %.2f' % r2)
print('MSE: %.2f'%mse)
""" 
XGBoost: 
R-squared: 0.65
MSE: 1932.91
XGBoost achives an R-squared of 0.65 and an MSE of 1932.9. This is the best performance out of all the boosting method we did test and implement.

Furthermore, we did not fine tune any of its parameters which further displays its modelling power. """