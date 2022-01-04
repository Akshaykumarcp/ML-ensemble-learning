# Table of content

1. Ensemble Learning
2. Bias & Variance
3. Methods of ensemble learning
    - 3.1. Generative method
        - 3.1.1. Bagging
        - 3.1.2. Boosting
    - 3.2. Non-Generative method
        - 3.2.1. Voting
        - 3.2.2. Stacking
4. Challenges in ensembles
5. Non-generative method: Majority Voting
    - 5.1. Hard Voting
    - 5.2. Soft Voting
6. Non-generative method: Stacking
    - 6.1 Intro to meta-learning
    - 6.2 Meta-learner approach
    - 6.3 Selecting base learners & meta learner
    - 6.4 Important characteristic of meta-learners
    - 6.5 Summary
7. Generative method: Bagging
    - 7.1 Creating base learners for bagging
    - 7.2 Summary
8. Generative method: Boosting
    - 8.1 Gradient boosting
    - 8.2 AdaBoost
        - 8.2.1 Introduction to adaboost
        - 8.2.2 Weighted Sampling
        - 8.2.3 Creating ensemble
        - 8.2.4 Strenghts of adaboost
        - 8.2.5 Weaknesses of adaboost
        - 8.2.6 Summary
9. Random Forest
    - 9.1 Intro to random forest
    - 9.2 Understanding random forest tree
    - 9.3 Summary


# 1 Ensemble Learning
- Ensemble learning involves a combination of techniques that allows multiple machine learning models called base learners to consolidate their predictions and output a single, optimal prediction, given their respective inputs and outputs.

# 2 Bias & Variance

- Machine learning models are not perfect, they are prone to a number of errors.
- Two most common sources of errors are:
    - bias and 
    - variance
- Ensemble learning aims to solve the problems of bias and variance.
- By combining many models, we can reduce the ensemble's error, while retaining the individual models complexities.
- Identify bias and variance using programs 0.0.1 & 0.0.2 present in the repo.
- High bias models usually have difficulty performing well in-sample. Thus is also called underfitting. It is due to the models simplicity or lack of complexity
- High variance models usually have difficulty generalizing or performing well out-of-sample, while they perform reasonably well in-sample. This is called overfitting. It is usually due to models unnecessary complexity.

# 3 Methods of ensemble learning
- Generative
    - Able to generate and affect the base learners that they use.
    - They can either tune thier learning algorithm or the dataset used to train them, in order to ensure diversity and high model performance.
    - 3.1 Generative methods are:
        - 3.1.1 Bagging
            - Bagging aims to reduce variance.
            - Bagging algorithm resamples instances of the training dataset, creating many individual and diverse datasets, originating from the same dataset.
            - Later, a separate model is trained on each samples dataset, forcing diversity between the ensemble models.
            Finally random forests, is similar to bagging, in that it resamples from the training dataset. Instead of sampling instances, it samples features, thus creating even more diverse trees as features strongly correlated to the target may be absent in many trees.
        - 3.1.2 Boosting
            - Boosting is a technique mainly targetting biased models
            - Main idea is to sequentially generate models such that each new model addresses biases inherent in the previous models.
            - Thus, by iteratively correcting previous errors, the final ensemble has significantly lower bias.
- Non-generative
    - Focused on combining the predictions of a set of pretrained models
    - Models are usually trained independently of one another and ensemble algorithm dictates how their predictions will be combined
    - Base classifiers are not affected by the fact that they exist in an ensemble
    - 3.2 Non-generative methods are:
        - 3.2.1 Voting
            - As the name implies, refers to techniques that allow models to vote in order to produce a single answer. 
            - Most popular answer is selected as the winner
        - 3.2.2 Stacking
            - Refers to methods that utilize a method (meta learner) that learns how to best combine the base learners predictions.
            - Stacking entails the generation of a new model, it does not affect the base learners, thus it is a non-generative method

# 4. Challenges in ensembles
Ensemble learning can greatly increase the performance of machine learning models, it comes at a cost. There are difficulties and drawbacks in correctly implementing it.
- Weak and noisy data
    - Important ingredient of successful model is the dataset
    - When data contains noise or incomplete information, there is not a single ML technique that will generate a highly perfoamance model
    - Adding more features to the dataset can greatly improve the models performance.
    - Adding more models to an ensemble cannot improve performance
- Understanding interpretability
    - By employing a large number of models, interpretability is greatly reduced
    - Example: a single decision tree can easily explain how it produced a prediction, by simply follwoing the decisions made at each node. On the other hand, it is difficult to interpret why an ensemble of 1k trees predicted a single value.
- Computational Cost
    - Training a single neural network is computationally expensive. Traiining a 1k of them requires a 1k times more computational resources
    - Some methods are sequential by nature. This means that it is not possible to harness the power of distributed computing. Instead, each new model must be trained when a previous model is completed
    - Imposes time penalties on the models development process, on top of the increased computational cost
    - When ensemble is put into production, the inference time will suffer as well.
- Choosing the right model
    - Models that comprise the ensemble must possess certain characteristics.
    - There is no point in creating any ensembe from number of identical models
    - Models achievable diversity depends on a number of factors, such as the size and quality of that dataset and the learning algorithm itself

# 5 Non-generative method: Majority Voting
- Majority Voiting is the simplest ensemble learning technique that allows combination of multiple base learner's predictions.
- The algorithm assumes that each base learner is a voter and each class is a contender.
- The algorithm takes votes into consideration in order to elect a contender as the winner
- Two main approaches to combining multiple predictions with voting:
    - 5.1. Hard Voting
        - Combines a number of predictions by assuming that most voted class in the winner
        - In a simple case of two classes and three base learners, if a target class has at least two votes, it becomes the ensemble's output
    - 5.2. Soft Voting
        - Soft voting takes into account the probability of the predicted classes.
        - In order to combine the predictions, soft voting calculates the average probability of each class and assumes that the winner is the class with the highest average probability. 
        - In the simple case of 3 base learners and two classes, we must take into consideration the predicted probability for each class and average them across the 3 learners.
        - In order for soft voting to be more effective than hard voting, the base classifiers must produce good estimates regarding the probability that a sample belongs to a specific class. 
        - If probabilities are meaningless (for ex: if they are always 100% for one class and 0 % for all others), soft voting could be even worse than hard voting

# 6 Non-generative method: Stacking

## 6.1 Intro to meta-learning
- Meta-learning is a broad ML term.
- Generally means: entails utilizing metadata for a specific problem in order to solve it
- Its application range from solving a problem more efficiently, to designing entirely new learning algorithms.
- It is a growing research field that has recently yielded impressive results by designing novel deep learning architectures.

Stacking is a form of meta-learning.
- The main idea is that we use base learners in order to generate metadata for the problems dataset and then utilize anaother learner called meta-learner.
- In order to process the meta-data, base learners are considered to be level 0 learners, while the meta learner is considered a level 1 learner. 
- In other words, the meta learner is stacked on top of base learners, hence the name stacking

## 6.2 meta-learner approach 1:

- We need metadata in order to both train and operate our ensemble.
- During the operation phane, we simple pass the data from our base learners. 
- On the other hand, the training phase is little more complicated. 
- We want our meta-learners to discover strengths and weaknesses between our base learners. 
- Although some would argue that we could train the base learners on the train set, predict on it, and use the predictions in order to train our meta-learners, this would induce variance.
- Our meta-learner would discover the strengths and weaknesses of data that has already been seen (by base learners).
- As we want to generate models with descent predictive (out-of-sample) performance, instead of descriptive (in-sample) capabilities, another approach must be utilized.

## 6.2 meta-learner approach 2:

- Another approach would be to split our training set into a base learner train set and a meta-learner train (validation) set.
- This way, would still retain a true set where we can measure the ensembles performance.
- Drawback of this approach is that we must donate some of the instances to  the validation set. 
- Further more, both the validation set size and the train set size will be smaller than the original train set size.
- Thus, the preferred approach is to utilize K-Fold CV.
- For each K, the base learners will be trained on K-1 folds and predict on the kth fold, generating 100/K percent of the final meta-data. 
- By repeating the process K times, one for each fold, we will have generated metadata for the whole training dataset

## 6.3 Selecting base learners & meta learner

- We described stacking as an advanced form of voting.
- Similarly to voting, stacking is dependent on the diversity of its base learners.
- If the base learners exhibit the same characteristics and performance throughout the problem's domain, it will be difficult for the meta-learner to dramatically improve their collective performance.
- Furthermore, a complex meta-learner will be needed.
- If the base learners are diverse and exhibit different performance characteristics in different domains of the problem, even a simple meta-learner will be able to greatly improve their collective performance.
- In general a good idea to mix different learning algorithms, in order to capture both linear and non-linear relationships between the features themselves, as well as the target variable.
- Take, for example, the following dataset, which exhibits both linear and non-linear relationships between the feature (x) and the target variable (y).
- It is evident that neither a single linear nor a single non-linear regression will be able to fully model the data.
- A stacking ensemble with a linear and non-linear regresion will be able to greatly outperform either of the two models.
- Even without stacking by hand crafting a simple rule (ex: "use the linear model if x is in the spaces [0,30]  or [60,100], else use the non-linear") we greatly outperform the two models.
- Generally, the meta-learning should be a relativelt simple ML algorithm, in order to avoid overfitting.
- Furthermore, additional steps should be taken in order to regularize the meta-learner.
- For ex: if a dicision tree is used, then the tree's max depth should be limited.
- If a regression model is used, a regularized regression (such as elastic net or ridge regression) should be preferred.
- if there is a need for a complex models in order to increase the ensemble's performance, a multi-level stack could be used, in which the number of models and each individual models complexity reduces as the stack's level increases.

## 6.4 Important characteristic of meta-learners
- Another really important characteristic of meta-learner should be the ability to handle correlated inputs and especially to not make any assumptions about the independence of features from one another, as naive bayes classifiers do.
- The inputs to the meta-learner (metadata) will be highly correlated.
- This happens because all the base learners are trained to predict the same target.
- Thus, their predictions will come from an approximation of the same function.
- Although the predicted values will vary, they will be close to each other

## 6.5 Stacking Summary
- Stacking can consist of many levels.
- Each level generates metadata for the next.
- You should create each level's metadata by splitting the train set into K folds and iteratively train on K-1 folds, while creating metadata for Kth fold.
- After creating the metadata, you should train the current level on the whole train set.
- Base learners must be diverse. The meta-learner should be realatively simple algorithm that is resistant to overfitting.
- If possible, try to induce regularization in the meta-learner.
- For ex: limit the maximum depth if you use a decision tree or use a regularized regression.
- The meta-learner should be able to handle correlated inputs relatively well.
- You should not be afraid to add under-performing models to the ensemble, as long as they introduce new information to the metadata (i,e they hanlde the dataset differently from the other models)

# 7 Generative method: Bagging

- Bagging A.K.A boostrap aggregating
- It can be a useful tool to reduce variance as it creates a number of base learners by sub sampling the original train set.

## 7.1 Creating base learners for bagging

- Bagging applies bootstrap sampling to the train set, creating a number of N bootstrap samples.
- Then creates the same number N of base learners, using the same machine learning algorithm.
- Each base learner is trained on the corresponding train set and all the base learners are combined by voting (hard voting for classification and averaging for regression)
- By using bootstrap samples with the same size as the original train set, each instance has a probability of 0.632 of appearing in any given bootstrap sample. 
- Thus, in many cases, this type of bootstrap estimate is referred to as the 0.632 bootstrap estimate.
- In our case, this means that we can use the remaining 36.8% of the original train set in order to estimate the individual base learners performance.
- This is called the out-of-bag score
- The 36.8% of instances are called out-of-bag instances.

## 7.2 Bagging Summary
- Bootstrap samples are created by resampling with replacement from the original dataset.
- The main idea is to treat the original sample as the population, and each subsample as an original sample.
- If the original dataset and the bootstrap dataset have the same size, each instance has a probability of 63.2% of being included in the bootstrap dataset (sample).
- Bootstrap methods are usefull for calculating stats such as confidence intervals and standard error, without making assumptions about the underlying distribution.
- Bagging generates a number of bootstrap samples to train each individual base learner. 
- Bagging benefits unstable learners, where small variations in the train set induce great variations in the generated model.
- Bagging is a suitable ensemble learning method to reduce variance
- Bagging allows for easy parallelization, as each bootstrap sample and base learner can be generated, trained and tested individually.
- As with all ensemble learning methods, using bagging reduces the interpretability and motivation behind individual predictions.

# 8 Generative method: Boosting

- Boosting aims to combine a number of weak learners into a strong ensemble.
- It is able to reduce bias, but also variance.
- Here, weak learners are individual models that perform slightly better than random.
- For example, in a classification dataset with two classes and an equal number of instances belonging to each class, a weak learner will be able to classify the dataset with an accuracy of slightly more than 50%. 

## Lets look at the 2 classic boosting algorithms:
- 8.1 Gradient boosting and
- 8.2 AdaBoost

## 8.2.1 Introduction to adaboost

- AdaBoost is one of the most popular boosting algorithm.
- Similar to bagging, the main idea behind the algorithm is to create a number of uncorrelated weak learners and then combine their predictions.
- The main difference with bagging is that instead of creating a number of independent bootstrapped train sets, the algorithm sequentially trains each weak learner, assigns weights to all instances, samples the next train set based on the instance's weights, and repeats the whole process.
- As a base learner algorithm, using decision trees consisting of a single node are used.
- These decision trees, with a depth of a single level are called decision stumps.

## 8.2.2 Weighted Sampling

- Weighted sample is the sampling process were each candidate has a corresponding weight, which determines its probability of being sampled.
- The weights are normalized, in order for their sum to equal one.
- then the normalized weights correspond to  the probability that any individual will be sampled.

## 8.2.3 Creating ensemble

Assuming a classification problem, the adaboost algorithm can be described on a high-level basis, from its basic steps.

For regression purpose, the steps are similar:
1. Initialize all of the train set instances weights equally, so their sum equals 1.
2. Generate a new set by sampling with replacement, according to the weights.
3. Train a weak learner on the sampled set.
4. Calculate its error on the original train set.
5. Add the weak learner to the ensemble and save its error rate.
6. Adjust the weights, increasing the weights of misclassified instances and decresing the weights of correctly classified instances.
7. Repeat from step 2
8. The weak learners are combined by voting. Each learners vote is weighted, according to its error rate.

## 8.2.4 Strenghts of adaboost

- Boosting algorithms are able to reduced both bias and variance
- For a long time, they were considered immune to overfitting, but in fact they can overfit, although they are extremely robust.
- One possible explanation is that the base learners, in order to classify outliers, create very strong and complicated rules that rarely fit other instances.

## 8.2.5 Weaknesses of adaboost
- One disadvantage of many boosting algorithms is that they are not easily parallelized, as the models are created in a sequential fashion.
- Furthermore, they pose the usual problems of ensemble learning techniques, such as reduction in interpretability and additional computaional costs.

## 8.2.6 Boosting Summary

- AdaBoost creates a number of base learners by employing weak learners (slightly better than random guessing)
- Each new base learner is trained on a weighted sample  from the original train set.
- Weighted sampling from a dataset assigns a weight to each instance and then samples from the dataset, using the weights in order to calculate the probabiliy that each instance will be sampled.
- The data weights are calculated based on the previous base learners errors.
- The base learners error is also used to calculate the learners weight.
- The base learners predictions are combined through voting, using each learners weight.
- Gradient boosting builds its ensemble by trianing each new base learner using the previous predictions errors as a target.
- The initial prediction is the train datasets target mean.

- Boosting methods cannot be parallelized in the degree that bagging methods can be.
- Althrough robust to overfitting, boosting methods can overfit.
- In scikit-learn, adaboost implementations store the individual learners weights, which can be used to identify the point where additional base learners do not contribute to the ensembles predictive power.
- Gradient boosting implementations store the ensembles error at each step (base learner), which can also help to identify an optimal number of base learners.
- XGBoost is a library dedicated to boosting, with regulaization capabilities that further reduce the overfitting ability of the ensembles.
- XGBoost is frequently a part of winning ML models in many kaggle competitions.

# 9 Random Forest

## 9.1 Intro to random forest

- Bagging is generally used to reduce variance of a model. 
- It achieves it by creating an ensemble of base learners, each one trained on a unique bootstrap sample of the original train set. 
- This forces diversity between the base learners. 
- Random Forests expand on bagging by inducing randomness not only on each base leamer's train samples, but in the features as well.
- Furthermore, their performance is similar to boosting techniques, although they do not require as much fine-tuning as boosting methods.

In this example, we will provide the basic background of random forests, as well as discuss the strengths and weaknesses of the method. Finally, we will present usage examples, using the scikit-learn implementation.

## 9.2 Understanding random forest tree

- In this example, we will go over the methodology of building a basic random forest tree. There are other methods that can be employed, but they all strive to achieve the same goal: diverse trees that serve as the ensemble's base learners.

- As mentioned earlier, create a tree by selecting at each node a single feature and split point, such that the train set is best split. When an ensemble is created, we wish the base learners to be as uncorrelated (diverse) as possible.

- Bagging is able to produce reasonably uncorrelated trees by diversifying each tree's train set through bootstrapping. 
- But bagging only diversifies the trees by acting on one axis: each set's instances.
- There is still a second axis on which we can introduce diversity, the features. 
- By selecting a subset of the available features during training, the generated base learners can be even more diverse. 
- In random forests, for each tree and at each node, only a subset of the available features is considered when choosing the best feature/split point combination. 
- The number of features that will be selected can be optimized by hand, but one-third of all features for regression problems and the square root of all features are considered to be a good starting point.

The algorithm's steps are as follows:

Select the number of features m that will be considered at each node For each base learner, do the following:

1. Create a bootstrap train sample
2. Select the node to split
3. Select m features randomly
4. Pick the best feature and split point from m
5. Split the node into two nodes
6. Repeat from step 2-2 until a stopping criterion is met, such as maximum tree depth

- Another method to create trees in a Random Forest ensemble is Extra Trees (extremely randomized trees). 
- The main difference with the previous method is that the feature and split point combination does not have to be the optimal. 
- Instead, a number of split points are randomly generated, one for each available feature. The best split point of those generated is selected

The algorithm constructs a tree as follows:

Select the number of features m that will be considered at each node and the minimum number of samples n in order to split a node

For each base learner, do the following:
1. Create a bootstrap train sample
2. Select the node split (the node must have at least n samples)
3. Select m features randomly
4. Randomly generate m split points, with values between the minimum and maximum value of each feature
5. Select the best of these split points
6. Split the node into two nodes and repeat from step 2-2 until there are no available nodes

## 9.3 Summary of random forest
- In this section, we discussed Random Forests, an ensemble method utilizing decision trees as its base learners. 
- We presented two basic methods of constructing the trees: 
        - the conventional Random Forests approach, where a subset of features is considered at each split
        - Extra Trees, where the split points are chosen almost randomly. 
        
- We discussed the basic characteristics of the ensemble method. Furthermore, we presented regression and classification examples using the scikit-learn implementations of Random Forests and Extra Trees.

- Random Forests use bagging in order to create train sets for their base learners. 
- At each node, each tree considers only a subset of the available features and computes the optimal feature/split point combination. - The number of features to consider at each point is a hyper-parameter that must be tuned. Good starting points are as follows:

    - The square root of the total number of parameters for classification problems
    - One-third of the total number of parameters for regression problems

- Extra trees and random forests use the whole dataset for each base learner. 
- In extra trees and random forests, instead of calculating the optimal feature/split-point combination of the 
        feature subset at each node, a random split point is generated for each feature in the subset and the 
        best is selected. 
- Random forests can give information regarding the importance of each feature. 
- Although relatively resistant to overfitting, random forests are not immune to it. 
- Random forests can exhibit high bias when the ratio of relevant to irrelevant features is low. 
- Random forests can exhibit high variance, although the ensemble size does not contribute to the problem.


