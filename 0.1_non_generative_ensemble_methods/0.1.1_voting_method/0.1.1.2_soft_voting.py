""" 
SOFT VOTING
- Scikit-learn's implementation allows for soft voting as well. 
- The only requirement is that the base learners implement the predict_proba function. 
- In our example, Perceptron does not implment the function at all, while SCV only produces probabilitie when 
        it is passeed the probability=True argument.
- Having these limitations in mind, we swap our Perceptron with a Naive Bayes Classifier implemented in the 
        sklearn.naive_bayes package.
- To acctually use soft voting, the VotingClassifier object must be initialized with the voting = 'soft' argument
"""

# Step 1: Import the required libraries

from sklearn import datasets, naive_bayes, svm, neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Step 2: Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

""" 
- We did instantiate the base learners and voting classifier. We will use Gaussian Naive Bayes implemented as 
        GaussianNB.
- Note that we will use probability=True in order for GaussianNB object to be able to produce probabilities
"""

# Step 3: Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = naive_bayes.GaussianNB()
learner_3 = svm.SVC(gamma=0.001, probability=True)
# Step 4: We will instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('NB', learner_2),
                           ('SVM', learner_3)],
                            voting='soft')

""" 
- We fit both VotingClassifier and the individual learners. 
- We want to analyze our results and as mentioned earlier, the classifier will not fit the objects that we pass 
        as arguments. Thus we have to manually fit our learners.
"""

# Step 5: We will fit classifier with the training data
voting.fit(x_train, y_train)
learner_1.fit(x_train, y_train)
learner_2.fit(x_train, y_train)
learner_3.fit(x_train, y_train)
""" SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
    
KNeighborsClassifier()
GaussianNB()
SVC(gamma=0.001, probability=True) """

# Step 6: We will predict the most probable class
hard_predictions = voting.predict(x_test)

# Step 7: We will get the base learner predictions
predictions_1 = learner_1.predict(x_test)
predictions_2 = learner_2.predict(x_test)
predictions_3 = learner_3.predict(x_test)

""" Finally we will print the accruracy of each base learner and the soft voting ensemble's accuracy """

# Step 8: print the accruracy of each base learner and the soft voting ensemble's accuracy
# Accuracy of base learners
print('L1:', accuracy_score(y_test, predictions_1))
print('L2:', accuracy_score(y_test, predictions_2))
print('L3:', accuracy_score(y_test, predictions_3))
""" 
L1: 0.94
L2: 0.96
L3: 0.88 """

# Accuracy of hard voting
print('-'*30)
print('Soft Voting:', accuracy_score(y_test, hard_predictions))
""" 
------------------------------
Soft Voting: 0.94 """