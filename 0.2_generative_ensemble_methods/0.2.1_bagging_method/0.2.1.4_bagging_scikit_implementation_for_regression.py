""" 
- For regression purposes, we will use the BaggingRegressor class from the same sklearn.ensemble package. 
- We will also instantiate a single DecisionTreeRegressor to compare the result. """

# Step 1: Import libraries and data
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
import numpy as np

diabetes=load_diabetes()

np.random.seed(1234)

# Split the train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

""" 
- We need to instantiate the single decision tree and the ensemble. 

- Note that we allow for a relatively deep decision tree by specifying max_depth = 6. 
- This allows the creation of diverse and unstable models, which greatly benefits bagging. 
- if we restrict the maximum depth to 2 or 3 levels, 
- we will see that bagging does not perform better than a single model.
 - Training and evaluating the ensemble and the model follows the standard procedure. """

# Step 2: Create the ensemble and a single base learner for comparison
estimator = DecisionTreeRegressor(max_depth=6)
ensemble= BaggingRegressor(base_estimator=estimator, n_estimators=10)

# Step 3: train and evaluate both the ensemble and the base learner
ensemble.fit(train_x, train_y)
ensemble_predictions = ensemble.predict(test_x)

estimator.fit(train_x, train_y)
single_predictions = estimator.predict(test_x)

ensemble_r2 = metrics.r2_score(test_y, ensemble_predictions)
ensemble_mse = metrics.mean_squared_error(test_y, ensemble_predictions)

single_r2 = metrics.r2_score(test_y, single_predictions)
single_mse = metrics.mean_squared_error(test_y, single_predictions)

# Step 4: Print the metrics
print('Bagging r-squared: %.2f'%ensemble_r2)
print('Bagging MSE: %.2f'% ensemble_mse)
print('-'*30)
print('Decision Tree r-squared: %.2f'%single_r2)
print('Decision Tree MSE: %.2f'%single_mse)

""" 
Bagging r-squared: 0.52
Bagging MSE: 2677.50
------------------------------
Decision Tree r-squared: 0.15
Decision Tree MSE: 4733.35

- The ensemble greatly outperform the single model by predicing both higher R-squared and lower mean 
        squared error (MSE). 
        As mentioned earlier, this is due to the fact that the base learners are allowed to create deep and 
        unstable models.

Exercise:
- change the max_depth (which is now 6) to be higher than 6 and lower than 6. 
                Note the result and compare them with mine to see is it better or more worse. """