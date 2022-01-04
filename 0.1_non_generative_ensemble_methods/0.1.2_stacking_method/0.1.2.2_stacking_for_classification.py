""" 
STACKING
- Stacking is a viable method for both regression and classification. In this example we will use it to classify the
        breast cancer dataset. Again, we will use three base learners:
        - a 5-neighbor k-NN,
        - a decision tree limited to a max depth of 4 and 
        - a simple neural network with 1 hidden layer of 100 neurons.

- For the meta learner, we will use a simple logistic regression """

# Step 1: Import libraries and data
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

bc = load_breast_cancer()
# Split train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

""" 
- We did instantiate the base learners and the meta-learner. 
- Note that MLPCLassifier has a hidden_layer_sizes = (100,) parameter, which specifies the number of neurons for 
        each hidden layer. 
        Here we will have a single layer of 100 neurons
"""

# Step 2: Create the ensemble's base learners and meta learner
# Append the base learners to a list for ease of access
base_learners = []

knn = KNeighborsClassifier(n_neighbors=2)
base_learners.append(knn)

dtr = DecisionTreeClassifier(max_depth=4, random_state=123456)
base_learners.append(dtr)

mlpc = MLPClassifier(hidden_layer_sizes =(100, ), solver='lbfgs', random_state=123456)
base_learners.append(mlpc)

base_learners
# [KNeighborsClassifier(n_neighbors=2), DecisionTreeClassifier(max_depth=4, random_state=123456), MLPClassifier(random_state=123456, solver='lbfgs')]

meta_learner = LogisticRegression(solver='lbfgs')
""" 
- Again using KFolds, we split the train set into 5 folds in order to train on four folds and generate metadata for 
        the remaining fold, repeated 5 times.

- Note that we use learner.predict_proba(train_x[test_indices])[:,0] in order to get the predicted probability that 
        the instance belongs to in the first class. 
        Given that we have only 2 classes, this is sufficient. 
        For N classes, we would have to either save N-1 features or simply use learner.predict, in order to save the 
        predicted class
"""

# Step 3: create the training metadata
# Create variables to store metadata and their targets
meta_data = np.zeros((len(base_learners), len(train_x)))
meta_targets = np.zeros(len(train_x))

# Create the cross-validation folds
KF = KFold(n_splits=5)
meta_index = 0
for train_indices, test_indices in KF.split(train_x):
    # Train each learner on the K-1 folds and create meta data for the Kth fold
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(train_x[train_indices], train_y[train_indices])
        predictions = learner.predict_proba(train_x[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = train_y[test_indices]
    meta_index += len(test_indices)

""" 
KNeighborsClassifier(n_neighbors=2)
DecisionTreeClassifier(max_depth=4, random_state=123456)
MLPClassifier(random_state=123456, solver='lbfgs')
KNeighborsClassifier(n_neighbors=2)
DecisionTreeClassifier(max_depth=4, random_state=123456)
MLPClassifier(random_state=123456, solver='lbfgs')
KNeighborsClassifier(n_neighbors=2)
DecisionTreeClassifier(max_depth=4, random_state=123456)
D:\anaconda\envs\ML\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:500: ConvergenceWarning: 
        lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
MLPClassifier(random_state=123456, solver='lbfgs')
KNeighborsClassifier(n_neighbors=2)
DecisionTreeClassifier(max_depth=4, random_state=123456)
DecisionTreeClassifier(max_depth=4, random_state=123456)
D:\anaconda\envs\ML\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:500: ConvergenceWarning: 
        lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
MLPClassifier(random_state=123456, solver='lbfgs') """

meta_data
""" 
array([[1.        , 1.        , 1.        , ..., 0.        , 0.5       ,
        0.        ],
       [1.        , 1.        , 1.        , ..., 0.01851852, 0.01851852,
        0.01851852],
       [1.        , 0.99999777, 0.9999912 , ..., 0.00105365, 0.00442989,
        0.03485871]]) """

meta_targets
""" 
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1.,
       1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1.,
       1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1.,
       0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1.,
       1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.,
       1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1.,
       1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0.,
       1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
       1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
       1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0.,
       1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
       0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.,
       0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 1., 0.,
       1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
       1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1.,
       1., 0., 0., 1., 1., 1., 1., 1., 1.]) """

# Transpose the metadata to be fed into the meta-learner
meta_data = meta_data.transpose()
""" 
array([[1.        , 1.        , 1.        ],
       [1.        , 1.        , 0.99999777],
       [1.        , 1.        , 0.9999912 ],
       ...,
       [0.        , 0.01851852, 0.00105365],
       [0.5       , 0.01851852, 0.00442989],
       [0.        , 0.01851852, 0.03485871]]) """

""" 
- Then we will train the base classifiers on the train set and create metadata for the test set as well as evaluating 
        their accuracy with metrics.accruracy_score(test_y, leaner.predict(test_x))
"""

# Step 4: create the metadata for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []
for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(test_y, learner.predict(test_x))


    base_acc.append(acc)

base_acc
# [0.863905325443787, 0.8816568047337278, 0.9112426035502958]

test_meta_data = test_meta_data.transpose()

""" 
- Finally we fit the meta-learner on the train metadata, evaluate its performance on the test data and print both the 
        ensemble's and the individual learner's accuracy.
"""

# Step 5: fit the meta-learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(test_y, ensemble_predictions)
# Step 6: Print the result
print('Acc  Name')
print('-'*20)
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
print(f'{acc:.2f} Ensemble')

""" Acc  Name
--------------------
0.86 KNeighborsClassifier
0.88 DecisionTreeClassifier
0.23 MLPClassifier
0.91 Ensemble

- Here we can see that meta-learner helps us to reach 91% accuracy of the performance and it out-perform all 
        the base learner.

Exercise:
- Please try to use the learner.predict method to generate our metadata. And see what is the accuracy of 
        the ensemble compared with all the methods. And also compared with the method which we used.
"""