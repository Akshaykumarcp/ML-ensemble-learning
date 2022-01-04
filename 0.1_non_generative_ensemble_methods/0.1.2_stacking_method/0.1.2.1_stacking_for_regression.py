""" 
STACKING
- Although scikit-learn does implement most ensemble methods, stacking is not one of them. We will need to implement 
        custom stacking solutions for both regression and classification problems
- We will try to create a stacking ensemble for the diabetes regression dataset. 
        The ensemble will consist of:
        - a 5 neighbor K-Nearest Neighbors (k-NN)
        - a decision tree timited to a max depth of four, and
        - a ridge regression (a regularized form of least squares regression).
- The meta-learner will be a simple ordinary Least Squares (OLS) linear regression.
- First we will need to import the libraries and data. 
        Scikit-learn provides a convenient method to split data into K-folds, with KFold class from the 
        sklearn.model_selection module. 
        We will use the first 400 instances for training and the remaining instances for testing.
"""

#Step 1: Import libraries and data
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

# load dataset
diabetes = load_diabetes()

# Split the traing set and testing set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

""" 
- In the next code, we will instantiate the base and meta-learners. 
- In order to have ease of access to the individual base learners later on, we will store each base learner in a
         list called base_learners. """

# Step 2: Create the ensemble's base learners and meta-learner and then append base learners to a list for ease of access
base_learners =[]
knn =  KNeighborsRegressor(n_neighbors=5)
base_learners.append(knn)
dtr = DecisionTreeRegressor(max_depth=4, random_state=123456)
base_learners.append(dtr)
ridge = Ridge()
base_learners.append(ridge)
meta_learner = LinearRegression()

base_learners
# [KNeighborsRegressor(), DecisionTreeRegressor(max_depth=4, random_state=123456), Ridge()]

""" 
- After instantiating our learners, we need to create the metadata for the training set. We split the training set 
        into 5 folds by first creating a KFold object, specifying the number of splits(K) with KFold(n_splits=5), 
        and then calling KF.split(train_x).

- This, in turn, returns a generator for the train and test indices of the five splits. For each of these splits, 
        we use the data indicated by train_indices (four folds) to train our base learners and create metadata on 
        the data corresponding to test_indices.

- Furthermore, we will store the metadata for each classifier in the meta_data array and the corressponding
        targets in the meta_targets array.

- Finally we transpose meta_data in order to get a (instance, feature) shape. """

# Step 3: Creating the training metadata
# Creating variables to store metadata and their targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_data
""" 
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]]) """

meta_targets = np.zeros(len(train_x))
""" 
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0.]) """

# Create the corss-validation folds
KF = KFold(n_splits=5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
  # Train each learner on the K-1 folds 
  # and create metadata for the Kth fold
  for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x[train_indices], train_y[train_indices])
    predictions = learner.predict(train_x[test_indices])
    meta_data[i][meta_index:meta_index+len(test_indices)] = \
                              predictions

  meta_targets[meta_index:meta_index+len(test_indices)] = \
                          train_y[test_indices]
  meta_index += len(test_indices)

# Transpose the metadata to be fed into the meta-learner
meta_data = meta_data.transpose()
meta_data
""" 
array([[221.        , 186.46031746, 179.44148461],
       [ 83.2       ,  91.72477064,  94.56884758],
       [134.4       , 186.46031746, 165.29144916],
       ...,
       [204.6       , 168.23076923, 160.66683682],
       [117.4       , 168.23076923, 156.86271927],
       [212.        , 168.23076923, 176.6069636 ]]) """

""" 
- For the test set, we do not need to split it into folds. We simply train the base learners on the whole train set
        and predict on the test set.

- Furthermore, we will evaluate each base learner and store the evaluation metrics, in order to compare them with 
        the ensemble's performance. As this is a regression problem , we will use R-squared and Mean Squared Error 
        (MSE) as evaluation metrics
"""

# Step 4: Cteate the metadata for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_errors = []
base_r2 = []
for i in range(len(base_learners)):
  learner = base_learners[i]
  learner.fit(train_x, train_y)
  predictions = learner.predict(test_x)
  test_meta_data[i] = predictions

  err = metrics.mean_squared_error(test_y, predictions)
  r2 = metrics.r2_score(test_y, predictions)

  base_errors.append(err)
  base_r2.append(r2)

""" 
KNeighborsRegressor()
DecisionTreeRegressor(max_depth=4, random_state=123456)
Ridge() """

test_meta_data
""" 
array([[155.6       ,  73.2       , 154.2       , 193.8       ,
        170.8       , 249.6       , 106.6       , 158.        ,
        182.4       , 205.4       , 174.2       , 117.8       ,
        230.8       , 158.4       , 168.4       , 148.2       ,
        193.4       , 142.        , 168.6       , 111.        ,
        125.4       , 184.2       , 196.8       , 184.2       ,
        141.4       , 106.        , 227.8       , 126.2       ,
        288.6       ,  91.2       , 134.4       , 137.2       ,
        231.4       ,  87.6       , 120.        , 132.        ,
         84.2       , 156.        , 123.4       ,  93.8       ,
        173.2       ,  91.2       ],
       [143.67647059,  82.84507042, 174.71052632, 292.73333333,
        143.67647059, 292.73333333,  82.84507042, 203.8       ,
        132.56818182, 174.71052632, 132.56818182, 143.67647059,
        292.73333333,  82.84507042, 132.56818182, 174.71052632,
        203.8       , 203.8       , 110.5       , 105.70512821,
        105.70512821, 230.60869565, 234.90909091, 174.71052632,
        177.6       ,  82.84507042, 132.56818182, 105.70512821,
        234.90909091,  82.84507042, 105.70512821,  82.84507042,
        177.6       ,  82.84507042,  82.84507042, 105.70512821,
         82.84507042, 230.60869565, 241.5       , 105.70512821,
        177.6       , 110.5       ],
       [160.20569328, 108.21800247, 165.5407445 , 205.62960611,
        169.93358485, 216.58535843,  76.13742934, 170.76883564,
        188.15646419, 174.49638664, 161.57492078, 137.76625604,
        203.67642707, 130.91177392, 182.29261133, 168.13673795,
        195.44510367, 146.72042166, 129.1291794 , 104.93802487,
        155.6499644 , 187.59982027, 182.21001172, 164.21819337,
        168.22643774, 104.38383881, 178.09185985, 137.03479685,
        231.33786855, 118.74748935, 133.92928772, 133.00941   ,
        200.9872349 ,  96.3090116 , 141.77479341, 129.7895615 ,
         80.91329649, 178.97297884, 140.18350942, 141.03921933,
        181.39375446,  88.6532805 ]]) """

base_errors
# [2697.826666666667, 3142.4515724436746, 2564.7702065249086]

base_r2
# [0.5126934338686302, 0.4323811444514658, 0.5367273302988218]

test_meta_data = test_meta_data.transpose()
""" 
array([[155.6       , 143.67647059, 160.20569328],
       [ 73.2       ,  82.84507042, 108.21800247],
       [154.2       , 174.71052632, 165.5407445 ],
       [193.8       , 292.73333333, 205.62960611],
       [170.8       , 143.67647059, 169.93358485],
       [249.6       , 292.73333333, 216.58535843],
       [106.6       ,  82.84507042,  76.13742934],
       [158.        , 203.8       , 170.76883564],
       [182.4       , 132.56818182, 188.15646419],
       [205.4       , 174.71052632, 174.49638664],
       [174.2       , 132.56818182, 161.57492078],
       [117.8       , 143.67647059, 137.76625604],
       [230.8       , 292.73333333, 203.67642707],
       [158.4       ,  82.84507042, 130.91177392],
       [168.4       , 132.56818182, 182.29261133],
       [148.2       , 174.71052632, 168.13673795],
       [193.4       , 203.8       , 195.44510367],
       [142.        , 203.8       , 146.72042166],
       [168.6       , 110.5       , 129.1291794 ],
       [111.        , 105.70512821, 104.93802487],
       [125.4       , 105.70512821, 155.6499644 ],
       [184.2       , 230.60869565, 187.59982027],
       [196.8       , 234.90909091, 182.21001172],
       [184.2       , 174.71052632, 164.21819337],
       [141.4       , 177.6       , 168.22643774],
       [106.        ,  82.84507042, 104.38383881],
       [227.8       , 132.56818182, 178.09185985],
       [126.2       , 105.70512821, 137.03479685],
       [288.6       , 234.90909091, 231.33786855],
       [ 91.2       ,  82.84507042, 118.74748935],
       [134.4       , 105.70512821, 133.92928772],
       [137.2       ,  82.84507042, 133.00941   ],
       [231.4       , 177.6       , 200.9872349 ],
       [ 87.6       ,  82.84507042,  96.3090116 ],
       [120.        ,  82.84507042, 141.77479341],
       [132.        , 105.70512821, 129.7895615 ],
       [ 84.2       ,  82.84507042,  80.91329649],
       [156.        , 230.60869565, 178.97297884],
       [123.4       , 241.5       , 140.18350942],
       [ 93.8       , 105.70512821, 141.03921933],
       [173.2       , 177.6       , 181.39375446],
       [ 91.2       , 110.5       ,  88.6532805 ]]) """

""" 
- Now, we have the metadata for both train and test sets. Now we can train our meta-learner on the train set and 
        evaluate on the test set.
"""

# Step 5: We need to fit the meta-learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data,meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)
err = metrics.mean_squared_error(test_y,ensemble_predictions)
r2 = metrics.r2_score(test_y, ensemble_predictions)

err
# 2066.6293192982876

r2
# 0.6267061744563556

# Step 6: Print the results
print('ERROR R2 Name')
print('-'*20)
for i in range(len(base_learners)):
  learner = base_learners[i]
  print(f'{base_errors[i]:.1f} {base_r2[i]:.2f} {learner.__class__.__name__}')
print(f'{err:.1f} {r2:.2f} Ensemble')
""" ERROR R2 Name
--------------------
2697.8 0.51 KNeighborsRegressor
3142.5 0.43 DecisionTreeRegressor
2564.8 0.54 Ridge
2066.6 0.63 Ensemble

- From the above result, r-squared has improved by over 16% from the best base learner (ridge regression), 
        while MSE has improved nearly 20%. This is a considerable improvement.
 """
