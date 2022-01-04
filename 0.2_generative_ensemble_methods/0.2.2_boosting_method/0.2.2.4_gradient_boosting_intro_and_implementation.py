""" 
- Gradient boosting is another boosting algorithm. 
- It is more generalized boosting framework compared to AdaBoosting, which also makes it more complicated and 
        math-intensive.
        Instead of trying to emphasize problematic instances by assigning weights and resampling the dataset, 
        gradient boosting builds each base learner's errors. 
        Furthermore, gradient boositng uses decision trees of varying depths. 

- In this example , we will present gradient boosting without delving much into math involved.
        Instead we will present the basic concept and also implement it.

Creating the ensemble
- Following that, it creates a decision tree that tries to predict the pseudo-residuals. 
        By repeating this process, a number of times, the whole ensemble is created. 
        Similar to AdaBoost, gradient boosting assigns a weight to each tree. 
        Contrary to AdaBoost, this weight does not depend on the tree's performance. 
        Instead it is a constant term, which is called learning rate.
        And its purpose is to increase the ensemble's generalization ability , by restricting its over-fitting power. 
        There are 9 steps on this algorithm:
            1. Defining the learning rate (smaller than 1 ) and the ensemble's size
            2. Calculate the trian set's target mean
            3. Using the mean as a very simple initial prediction, calculate each instance's target difference 
                    from the mean. These errors are called pseudo-residuals
            4. Build a decision tree, by using the original train set's features and the pseudo-residulas as targets
            5. Make predictions on the train set, using the decision tree (we try to predict the pseudo-residuals)
            6. Multiply the predicted values by the learning rate
            7. Add the multiplied values to the previously stoed predicted values. Use the newly calculated values as 
                    predictions
            8. Calculate the new pseudo-residuals using the calcualated predictions
            9. Repeat from step 4 until the desited ensemble size is achieved.

- Note that in order to produce the final ensemble's predictions, each base learner's predictions is 
                    multiplied by the learning rate and added to the previous learner's prediction. 
                    The calculated mean can be regarded as the first base learner's prediction.
- At each steps for a learning rate lr, the prediction is calculated by this
        formula: ps = mean +lr.p1+lr.p2+....+lr*ps

- The residuals care calcualted as the difference from the actual target value t: r2 = t-ps

- Although gradient boosting can be complex and mathematical intensive, if we focus on conventional regression 
        problems it can be quite simple. We will implement it using a standard scikit-learn decision tree

 """

# Import the linbraries and data
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()
train_size = 400

#Split the train and test set
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

""" 
- Following this, we define the ensemble's size, learning rate and Decision Tree's maximum depth. 
- Furthermore, we will create a list to store individual base learners, as well as Numpy array to store the 
        previous predictions.

- As mentioned eariler, our initial prediction is the train set's target mean. Instead of defining a maximum depth, 
        we could also define a maximum number of leaf nodes by passing the max_leaf_nodes = 3 argument to the 
        constructor.
 """

# Step 2: Create the ensemble and define the ensemble's size, learning rate and decision tree depth
ensemble_size = 50
learning_rate = 0.1
base_classifier = DecisionTreeRegressor(max_depth=3)

# Create placeholders for the base learners and each step's prediction
base_learners = []

# Note that the initial prediction is the target variable's mean
previous_predictions = np.zeros(len(train_y)) + np.mean(train_y)

""" 
- The next step is to create and train the ensemble. 
- We start by calculating the pseudo-residuals using the previous predictions. 
- We then create a deep copy of the base learner class and train it on the ttrain set using the pseudo-residuals
        as targets.
 """
# Crteate the base learners
for _ in range(ensemble_size):
    # Start by calcualting the pseudo-residuals
    errors = train_y - previous_predictions

    # Make a deep copy of the base classifier and train it on the
    # pseudo-residuals
    learner = deepcopy(base_classifier)
    learner.fit(train_x, errors)

    # Predict the residuals on the train set
    predictions = learner.predict(train_x)
    # Multiply the predictions witht he learning rate and add the results to the previous prediction
    previous_predictions = previous_predictions + learning_rate* predictions
    # Save the base learner
    base_learners.append(learner)

"""
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)
DecisionTreeRegressor(max_depth=3)

- Finally we use the trained base learner in order to predict the pseudo-residuals on the train set. 
- We multiply the predictions by the learning rate and add them to the previous predictions. 
- Finally, we append the base learner to the base_learners list

- In order to make predictions with our ensemble and evaluate it, we use the test set's features on order to 
        predict pseudo-residuals, multiply them by the learning rate and dd them to the trainset's target mean. 
        It is important to use the original train set's mean as a starting point because each tree predicts the 
        deviations from that orginal mean
 """

# Step 3: Evaluate the ensemble
# Start with the train set's mean
previous_predictions = np.zeros(len(test_y)) + np.mean(train_y)

# for each base learner predict the pseudo-residuals for the test set and add them to the previous prediction,
#        after multiplying with the learning rate

for learner in base_learners:
  predictions= learner.predict(test_x)
  previous_predictions = previous_predictions+ learning_rate*predictions

# Step 4: Print the metrics
r2 = metrics.r2_score(test_y, previous_predictions)
mse = metrics.mean_squared_error(test_y, previous_predictions)
print('Gradient Bossting: ')
print('R-squred: %.2f '%r2)
print('MSE: %.2f' % mse)

""" 
Gradient Bossting: 
R-squred: 0.59 
MSE: 2255.44

- The algorithm is able to achieve an R-squred value 0f 0.59 and an MSE of 2255.44 with this particular setup. """