""" 
- Scikit-learn also implements random forests for regression purposes in teh RandomForestRegressor class. 
- It is also highly parameterizable with hyper-parameters concerning both the ensemble as a whole, as well as
                 individual trees.
- Here we will generate an ensemble inorder to model the diabetes regression dataset.
 """

# Step 1: Importing libraries and dataset
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

diabetes  =  load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

""" 
- Following this we will create the ensemble by setting the n_estimators and n_jobs parameters. 
- These parameters dictate the number of trees that will be generated and the number of parallel jobs that will be 
                run. We train the ensemble using the fit function and evaluate it on the test set by measuring its 
                achieved accuracy """

# Step 2: create the ensemble
ensemble_size = 100
ensemble = RandomForestRegressor(n_estimators=ensemble_size, n_jobs=4)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
# RandomForestRegressor(n_jobs=4)

# Step 4: Evaluate the ensemble
predictions = ensemble.predict(test_x)

# Step 5: Print the metrics
r2= metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('Random Forest: ')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)

""" 
Random Forest: 
R-squared: 0.51
MSE: 2729.81

- The ensemble is able to produce R-squared of 0.51 and an MSE of 2729.81 on the test set. 
- As the R-squared and MSE on the train set are 0.92 and 468.13 respectively, it is safe to assume that the 
                ensemble overfits.
- This is the case where the error limit overfits, and thus we need to regulate the individual trees in order to 
                achieve better results.
- By reducing the minimum number of samples required to be at each leaf node (increased to 20 from the default value of 2)
                 through min_samples_leaf = 20.
- We are able to increase R-squared to 0.6 and MSE to 2206.6
- Furthermore, by increasing the ensemble size to 1000, R-squared is further to increased to 0.61 and MSE is 
                further decreased to 2158.73 """