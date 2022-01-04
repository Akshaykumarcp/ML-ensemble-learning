# We will use the breast cancer classification dataset for this example

#Step 1: Import libraries and data
from copy import  deepcopy
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

bc = load_breast_cancer()

train_size = 400

#Split the train and test set
train_x, train_y = bc.data[:train_size], bc.target[:train_size]
test_x, test_y = bc.data[train_size:], bc.target[train_size:]
np.random.seed(123456)

""" 
- We then create the ensemble.
- First, we declare the ensemble's size and the base learner type. 
- As mentioned earlier, we use the decision stumps (decision trees only a single level deep).

- Furthermore, we create a numpy array for the data instance weights, the learners weights and the learners errors.
 """

# Step 2: Create the ensemble
ensemble_size = 3
base_classifier = DecisionTreeClassifier(max_depth=1)

# create the initial weights
data_weights = np.zeros(train_size) + 1/train_size

# Create a list of indices for the train set
indices = [x for x in range(train_size)]
base_learners = []
learners_errors = np.zeros(ensemble_size)
learners_weights = np.zeros(ensemble_size)

""" 
- For each base learner, we will create a deepcopy of the original classifier , train it on the sample 
        dataset and evaluate it.

- First we will create a copy and sample with replacement from the original test set according to the instance weights.
 """

# Create each base learner
for i in range(ensemble_size):
  weak_learner = deepcopy(base_classifier)
  #Choose the samples by sampling with replacement
  #Each instance's probability is dictated by its weight
  data_indices = np.random.choice(indices, train_size, p=data_weights)
  sample_x, sample_y = train_x[data_indices],train_y[data_indices]
  # Fit the weak learner and evaluate it
  weak_learner.fit(sample_x, sample_y)
  predictions = weak_learner.predict(train_x)
  errors = predictions != train_y
  corrects = predictions == train_y
  # calculate the wieghted errors
  weighted_errors = data_weights*errors
  # The base learners error is the average of the weighted errors
  learner_error = np.mean(weighted_errors)
  learners_errors[i] = learner_error
  #the learner's weight
  learner_weight = np.log((1-learner_error)/learner_error)/2
  learners_weights[i] = learner_weight
  # Update the data weights
  data_weights[errors] =  np.exp(data_weights[errors]*learner_weight)
  data_weights[corrects] =  np.exp(-data_weights[corrects]*learner_weight)
  data_weights = data_weights/sum(data_weights)
  #Save the learner
  base_learners.append(weak_learner)

""" 
- We then fit the learner on the sampled dataset and predict the original train set. 
- We use the predictions to see which instances are correctly classified and which instances are misclassified.

- The wieghted errors are classified . 
- Both errors and corrects are lists of Booleans (True or False), but Python handles them as 1 and 0. 
- This allows us to multiply element-wise with data_weights.

- Finally , the learner's weight can be calcuated as half the natural logarithm of the weighted accuracy 
        over the weighted error. In turn, we can use the learner's weight to calcuate the new data weights.

- For erroneously classified instances, the new weigth equals to the natural exponent of the old weight times 
        the learner's weight.

- For correctly clssified instance, the negative multiple is used instead.

- Finally the new weights are normalized and the base learner is added to the base_learners list

In order to make preedictions with the ensemble, we combine each individual predictions through a weighted majority 
        voting. 
        As this is a binary classificaion problem, if the weighted average is more than 0.5, the instance is 
        classified as 0; otherwise, it's classified as 1.
 """

# Step 3: Evaluate the ensemble
ensemble_predictions = []
for learner, weight in zip(base_learners, learners_weights):
    # Calculate the weighted predictions
    prediction = learner.predict(test_x)
    ensemble_predictions.append(prediction*weight)

# The final prediction is the weighted mean of the individual predictions
ensemble_predictions = np.mean(ensemble_predictions, axis=0) >= 0.5

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)
# Step 4: Print the accuracy
print('Boosting: %.2f' % ensemble_acc)
# Boosting: 0.95