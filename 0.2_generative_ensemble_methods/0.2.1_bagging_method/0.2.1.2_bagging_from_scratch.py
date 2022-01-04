""" 
- In this example, we will implement it using the decision tree. 
- We will try to classify the MNIST dataset of handwritten digits
- We will use 1500 instances as the train set and the remaining 297 as the test set.
- We will generate 10 bootstrap samples and consequently 10 decision-tree models. 
- Then we will combine the base predictions using hard-voting
 """

# Step 1: import libraries and load data
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

digits = load_digits()

train_size = 1500

# Split the train and test set
train_x, train_y =digits.data[:train_size], digits.target[:train_size]
test_x, test_y =digits.data[train_size:], digits.target[train_size:]

""" 
- We will create our bootstrap samples and train the models.

- Note that we do not use np.random.choice. 
- Instead we will generate an array of indices with np.random.randint(0,train_size, size = train_size), 
                as this will enable us to choose both the features and the corresponding targets for each 
                bootstrap sample. 
                We store each base learner in the base_learners list for the ease of access
 """

# Step 2: Create our bootstrap samples and train the classifiers
ensemble_size = 10
base_learners = []

for _ in range(ensemble_size):
  # We sample indices in order to access features and targets
  bootstrap_sample_indices = np.random.randint(0, train_size, size=train_size)
  bootstrap_x = train_x[bootstrap_sample_indices]
  bootstrap_y = train_y[bootstrap_sample_indices]
  dtree = DecisionTreeClassifier()
  dtree.fit(bootstrap_x, bootstrap_y)
  base_learners.append(dtree)

"""
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()
DecisionTreeClassifier()

- Next we will predict the targets of the test set with each base learner and store their preidctions as well as 
                their evaluated accuracy
 """

# Step 3: We will predict with the base learners and evaluate them

base_predictions = []
base_accuracy = []

for learner in base_learners:
    predictions = learner.predict(test_x)
    base_predictions.append(predictions)
    acc = metrics.accuracy_score(test_y, predictions)
    base_accuracy.append(acc)

# Now that we have each base learner's predictions in base_predictions, we can combine them with hard voting

# Step 4: Combine the base learner's predictions
ensemble_predictions = []
# Find the most voted class for each test instance
for i in range(len(test_y)):
    # Count the votes for each class
    counts = [0 for _ in range(10)]
    for learner_predictions in base_predictions:
        counts[learner_predictions[i]] = counts[learner_predictions[i]]+1

    # Find the class with most votes
    final = np.argmax(counts)
    # Add the class to the final predictions
    ensemble_predictions.append(final)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

""" 
- Finally we will print the accuracy of each base learner as well as the ensemble's accuracy, sorted in ascending order
 """
# Step 6: Print the accuracies
print('Base Learners: ')
print('-'*30)
for index, acc in enumerate(sorted(base_accuracy)):
  print(f'Learner {index + 1}: %.2f' %acc)
  print('-'*30)
  print('Bagging: %.2f'% ensemble_acc)
  
""" 
Base Learners: 
>>> print('-'*30)
------------------------------
>>> for index, acc in enumerate(sorted(base_accuracy)):
...   print(f'Learner {index + 1}: %.2f' %acc)
...   print('-'*30)
...   print('Bagging: %.2f'% ensemble_acc)
...
Learner 1: 0.70
------------------------------
Bagging: 0.87
Learner 2: 0.74
------------------------------
Bagging: 0.87
Learner 3: 0.75
------------------------------
Bagging: 0.87
Learner 4: 0.75
------------------------------
Bagging: 0.87
Learner 5: 0.75
------------------------------
Bagging: 0.87
Learner 6: 0.76
------------------------------
Bagging: 0.87
Learner 7: 0.76
------------------------------
Bagging: 0.87
Learner 8: 0.77
------------------------------
Bagging: 0.87
Learner 9: 0.77
------------------------------
Bagging: 0.87
Learner 10: 0.79
------------------------------
Bagging: 0.87 

- It is eveident that the ensemble accuracy is almost 7% higher than the best-performing base model. 
- This is a considerable improvement, especiialy if we take into account that this ensemble consists of identical base 
                learners.
"""
