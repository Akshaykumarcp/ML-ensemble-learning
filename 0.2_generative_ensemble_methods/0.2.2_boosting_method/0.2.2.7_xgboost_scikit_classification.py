""" 
- For classification purposes, the correspoding class is implemented in XGBClassifier. 
        The constructor are the same as the regression implementation.
 """

# Step 1: Import libraries and data
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np

digits = load_digits()

train_size = 1500

train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

#Step 2: Create the ensemble
ensemble_size = 100
ensemble = XGBClassifier(n_estimators=ensemble_size, n_jobs=4)

# Step 3: Evaluate the ensemble
ensemble.fit(train_x,train_y)
""" XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=4,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1) """

#Step 4: Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)
ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

#Step 4: print the accuracy
print('XGBoost accuracy: %.2f' % ensemble_acc)
""" 
XGBoost accuracy: 0.90

- The ensemble correctly classifies the test set with 89% accuracy, also the higest achieved for any boosting algorithm """