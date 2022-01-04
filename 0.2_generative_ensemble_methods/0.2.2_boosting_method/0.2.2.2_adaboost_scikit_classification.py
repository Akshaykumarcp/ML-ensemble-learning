""" 
- Scikit-learn's Adaboost implementation exist in the sklearn.ensemble pakcage, in the AdaBoostClassifier and 
        AdaBoostRegressor classes.

- Like all scikit-learn classifiers, we will use the fit and predict functions in order to train the classifier 
        and predict on the test set. 
        The first parameter is the base classifier that the algorithm will use. 
        The algorithm="SAMME" parameter forces the classifier to use a discrete boosting algorithm.
        We will use the hand-written digits recognition.
 """
# Step 1: Import libraries and load data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

digits = load_digits()

#Split the train and test set
train_size = 1500

train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

# Step 2: Create the ensemble
ensemble_size = 200
ensemble = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=ensemble_size)

# Step 3: train the ensemble
ensemble.fit(train_x,train_y)
""" AdaBoostClassifier(algorithm='SAMME',
                   base_estimator=DecisionTreeClassifier(max_depth=1),
                   n_estimators=200) """

#Step 4: Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)
ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Step 5: Print the accuracy
print('Boosting: %.2f' % ensemble_acc)
# Boosting: 0.81

""" 
- This result in an ensemble with 81% accuracy on the test set. 
- One advantage of using the provided implementation is that we can access and plot each individual base learner's 
        errors and weights. We can access them through ensemble.estimator_errors_ and ensemble.estimator_weights_ 
        respectively.

- By plotting the weights we can gauage where the ensemble stops to benefit from additional base learners.

- By creating an ensemble of 1000 base learners, we can  see that from approximately the 200 base learners mark, 
        the weights are stabilized.

Exercise:
- implment AdaBoost for regression using the scikit-learn. 
- And also calcuate the R-squared and MSE for the implementation. 

- also use the diabetes dataset. 
Hint: choose the train_size = 400 """