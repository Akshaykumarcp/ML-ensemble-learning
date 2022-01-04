""" 
HARD VOTING
- The simplest way to implement hard voting in Python is to use scikit-learn to create base learners, train them on 
        some data, and combine their predictions on test data. In order to do it, there will be 6 steps:

        1. Load the data and split it into train and test sets
        2. Create some base learners
        3. We need to trian them on the train data
        4. Produce predictions for the test data
        5. Combine predictions using hard voting
        6. Compare the individual learner's predictions as well as the combined predictions with the ground truth 
                (actual correct classes)

- Although scikit-learn has implementations for voting, by creating a custom implementation, it will be easier to 
        understand how the algorithm works.

- Furthermore, it will enable us to better understand how to processs and analyze a base learner's outputs.

Custom hard voting implementation:
- To implement a custom hard voting solution, we will use three base learners: 
        - a Perceptron (a neural network with a single neuron)
        - a Support Vector Machine (SVM) and 
        - a Nearest Neighbor

- Furthermore, we will use the argmax function from Numpy. 
        This function will return the index of an array's (or array -like data structure) element with the highest value.

- Finally , accuracy_score will calculate the accuracy of each classifier on our test data. """

# Step 1: Import the libraries and data
from sklearn import datasets, linear_model, svm, neighbors
from sklearn.metrics import accuracy_score
from numpy import argmax

# Load the datasets
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

""" 
- We then instantiate our base learneres. We hand-picked their hyperparameters to ensure that they are diverse 
        in order to produce a well-performing ensemble.

- As breast_caner is a classification dataset, we will use SVC, the classification version of SVM, along with 
        KNeighborsClassifier and Perceptron.

- Furthermore, we set the random state of Perceptron to 0 to ensure the reproducibility of our example. """

# Step 2: Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

""" 
- We will split the data into train and test sets, using 100 instances for our test set and train our base learner's on 
        the train set.
"""

# Step 3: split the train and test samples
test_samples= 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Fit learners with the train data
learner_1.fit(x_train, y_train)
learner_2.fit(x_train, y_train)
learner_3.fit(x_train, y_train)
""" SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False) 

KNeighborsClassifier()
Perceptron(tol=0.01)
SVC(gamma=0.001)

- By storing each base learner's prediction in predictions_1, predictions_2, and predictions_3, we can further 
        analyze and combine them into our ensemble
"""

# Step 4: Each learner predicts the classess of the test data
predictions_1 = learner_1.predict(x_test)
predictions_2 = learner_2.predict(x_test)
predictions_3 = learner_3.predict(x_test)

""" 
- We combine the predictions of each base learner for each test instance.

- The hard_predictions list will contain the ensemble's predictions (output)

- By iterating over every test sample with for i in range(test_samples), we count the total number of votes that each 
        class has received from the 3 base learners.

- As the dataset contains only 2 classes, we need a list of two elements: counts = [0 for _ in range(2)]. 
        In step 3, we stored each base learner's predictions in an array.

- Each one of those array's elements containes the index of the instance's predicted class (in our case, 0 and 1).

- Thus we need to increase the corresponding element's value in counts[predictions_1[i]] by one to count the base
         learner's vote. Then argmax(counts) returns the element (class with the highest number of votes).
"""

# Step 5: We will need to combine the predictions with hard voting
hard_predictions = []
# for each predicted sample
for i in range(test_samples):
  #Count the votes for each class
  counts = [0 for _ in range(2)]
  print('iteration: ', i)
  counts[predictions_1[i]] = counts[predictions_1[i]] + 1
  counts[predictions_2[i]] = counts[predictions_2[i]] + 1
  counts[predictions_3[i]] = counts[predictions_3[i]] + 1
  print(counts)
  # Find the class with most votes
  final = argmax(counts)
  print('max: ',final)
  # Add the class to the final predictions
  hard_predictions.append(final)
  print('hard_predictions: ',hard_predictions)
""" 
iteration:  0
[0, 3]
max:  1
hard_predictions:  [1]
iteration:  1
[0, 3]
max:  1
hard_predictions:  [1, 1]
iteration:  2
[0, 3]
max:  1
hard_predictions:  [1, 1, 1]
iteration:  3
[2, 1]
max:  0
hard_predictions:  [1, 1, 1, 0]
iteration:  4
[0, 3]
max:  1
hard_predictions:  [1, 1, 1, 0, 1]
iteration:  5
[0, 3]
max:  1
hard_predictions:  [1, 1, 1, 0, 1, 1]
iteration:  6
[0, 3]
max:  1
hard_predictions:  [1, 1, 1, 0, 1, 1, 1]
iteration:  7
[3, 0]
max:  0
hard_predictions:  [1, 1, 1, 0, 1, 1, 1, 0]
iteration:  8
[0, 3]
max:  1
hard_predictions:  [1, 1, 1, 0, 1, 1, 1, 0, 1]
iteration:  9
[0, 3]
max:  1
hard_predictions:  [1, 1, 1, 0, 1, 1, 1, 0, 1, 1]

- Finally we will calcuate the accuracy of the individual base learners as well as the ensemble with accuracy_score 
        and print them on the screen """

# Step 6: Print the accuracies of base learners
print('L1: ', accuracy_score(y_test, predictions_1))
print('L2: ', accuracy_score(y_test, predictions_2))
print('L3: ', accuracy_score(y_test, predictions_3))
# Print the accuracy of hard voting
print('-'*30)
print('Hard Voting: ', accuracy_score(y_test,hard_predictions))
""" 
L1:  0.94
L2:  0.78
L3:  0.88
------------------------------
Hard Voting:  0.9

Analysing our results:
- We can visualize the learner's errors in order to examine why the ensemble performs in this specific way
 """

# Step 1: Import the required libraries
import matplotlib as mlp
import matplotlib.pyplot as plt
mlp.style.use('seaborn-paper')

""" 
- Then we need to calculate the errors by subtracting our prediction form the actual target.

- Thus we get -1 each time the learner predicts a positive (1) when the true class is negative (0), and a 1 when it 
        predicts a negative (0) while the true class is positive (1). If the prediction is correct, we get a zero (0)
 """
# Step 2: Calculating the errors
errors_1 = y_test-predictions_1
errors_2 = y_test-predictions_2
errors_3 = y_test-predictions_3

""" 
- For each base learner, we plot the instances where they have predicted the wrong class.

- Our aim is to scatter plot the x and y lists.

- These lists will contain the instance number (the x list) and the type of error (the y list).

- With plt.scatter, we can specify the coordinates of our points using the aforementioned lists, as well as specify how 
        these points are depicted.

- This is really important to ensure that we can simultaneously visualize all the errors of the classifiers as well as the 
        relationship between them.

- The default shape for each point is a circle.

- By specifying the marker parameter, we can alter this shape.

- Furthermore, with the s parameter we can specify the marker's size.

- Thus, the:
        - first learner will have around shape of size 120
        - second learner (Perceptron) will have an x shape of size 60 and 
        - third learner (SVM) will have a round shape of size 20.

- The if not errors_*[i] == 0 guard ensures that we will store correctly classified instances
 """

# Step 3: Discard correct predictions and plot each learner's errors
x = []
y = []
for i in range(len(errors_1)):
  if not errors_1[i] == 0:
    x.append(i)
    y.append(errors_1[i])
plt.scatter(x, y, s=120, label='Learner 1 Errors')

x = []
y = []
for i in range(len(errors_2)):
  if not errors_2[i] == 0:
    x.append(i)
    y.append(errors_2[i])
plt.scatter(x, y, marker='x', s=60, label='Learner 2 Errors')


x = []
y = []
for i in range(len(errors_3)):
  if not errors_3[i] == 0:
    x.append(i)
    y.append(errors_3[i])
plt.scatter(x, y, s=20, label='Learner 3 Errors')

#Finally, we need to specify the figure's title and labels and plot the legend
plt.title('Learner errors')
plt.xlabel('Test sample')
plt.ylabel('Error')
plt.legend()
plt.show()

""" 
- There are 10 samples where at least 2 learners predict the wrong class. 
- These are 10 cases out of 100 that the ensemble predicts wrong as the most voted class is wrong. 
- Thus, it produces a 90% accuracy. """