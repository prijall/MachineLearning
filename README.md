# This Repository will contain all my machine learning work!!!

# Day 1

Today, I continued learning Machine learning Algorithms. I learned about Naive Baye's algorith which is used to solve classification problems having many 
independent features.

![Naive Bayes](image_5_uhsgzr.webp)

# Day 2
Revised the concept of Simple Linear Regression through practical implementation with the help of Krish Naik's video. It was littele bit hard for me to grasp the concept of everything as I am a beginner but hope to keep learning and improving.

![Code Img](<Screenshot 2024-01-23 190500.png>)


# Day 3

The k-Nearest Neighbors (KNN) algorithm is a versatile and simple supervised machine learning algorithm used for both classification and regression tasks. It's a non-parametric and instance-based learning method, meaning it doesn't make explicit assumptions about the underlying data distribution and stores the entire training dataset in memory for prediction. Here's how KNN works:

### Basic Idea:

- Training Phase:
 In the training phase, KNN simply memorizes the entire training dataset. No actual "learning" occurs during this phase, which is why it's considered non-parametric.

- Prediction Phase:
 When a prediction is required for a new, unseen data point, KNN looks at the K-nearest neighbors of that data point within the training dataset. These nearest neighbors are identified based on a distance metric, typically Euclidean distance, but other metrics can be used.

- Voting (Classification) or Averaging (Regression):
 For classification tasks, KNN counts the class labels of the K-nearest neighbors and assigns the majority class as the prediction. For regression tasks, it averages the target values of the K-nearest neighbors to make the prediction.

 ### Key Parameters:

- K (Number of Neighbors): 
The most crucial hyperparameter in KNN is K, which determines how many neighbors are considered when making a prediction. A smaller K value may lead to a noisy prediction, while a larger K value may result in a smoother but potentially biased prediction.

- Distance Metric:
 The choice of distance metric affects how KNN calculates the similarity between data points. Common distance metrics include Euclidean distance, Manhattan distance, and Minkowski distance.

### Advantages of KNN:

- Simple and easy to understand.
- No model training involved during the training phase.
- Versatile: Suitable for both classification and regression tasks.
- Effective when the decision boundary is nonlinear or complex.

### Disadvantages of KNN:

- Memory Intensive: 
   KNN stores the entire training dataset, making it memory-intensive for large datasets.
- Computationally Expensive: 
   Calculating distances between data points can be computationally expensive, especially for high-dimensional data.
- Sensitive to the Choice of K: 
   The choice of K can significantly impact the model's performance. It requires careful tuning.
- Not Suitable for High-Dimensional Data: KNN's performance tends to degrade as the dimensionality of the data increases (curse of dimensionality).

### Use Cases:

- KNN is often used in recommendation systems, such as recommending products or movies based on user behavior.
- It's suitable for image classification tasks.
- In anomaly detection, KNN can identify outliers by considering data points with few nearby neighbors as anomalies.- 
- KNN can be used in text classification and natural language processing (NLP) tasks.
- Choosing the Right K: Selecting the appropriate K value is essential for the performance of the KNN algorithm. It involves experimentation and might require  cross-validation to determine the K that results in the best performance for your specific dataset. A small K value (e.g., 3 or 5) may capture noise, while a large K value may lead to oversmoothed predictions.

![KNN algo](KNN.png)


# Day 4

  Learning about Underfitting , Overfitting and generalized model is very important in Machine Learning. 

### Overfitting 
   When the best fit line completely fitted through all training data, then the modwl is said to be overfitted. In this case, model is
   - Low Biased
   - High variance

### Underfitting
   When the best fit line is distant from the training datasets then the model is said to be Underfitted. In this case, model is
   - High Biased
   - Can be low variance or high variance

### Generalized model
   When the best fit line is accurately/precisely fitted with training datasets then the model is considered to be Generalized. In this case, model is
   - Low Biased
   - Low variance.

   **Note:** Biasness is generally used for model with training dataset and Variance is used for model with testing dataset. 


# Day 5

  Today, I learnt the concept of Confusion matrix and accuracy performance parameter in Machine Learning Algorithm. It is one of the most used performance metrics in classification. 

### Confusion Matrix:
   It a table that shows the number of True Positive i.e. (1,1), True Negative i.e. (0,0), False Positive i.e. (1,0) and False Negative i.e. (0,1).Our aim is to reduce False Positive and False Negative values and increase True Positive and True Negative Values. 

### Accuracy:
   It measures the overall correctness of the model. The main disadvantage of accuracy as performance parameter iit doesn't work well with imbalance datasets.

   - Formula to calculate accuracy is:
      (TP + TN)/(TP + TN + FP + FN)


**Note**: In coordinate (x,y) used above, the value of x represents predicted values whereas the value of y represents actual values of the datasets.


# Day 6
  After learning about confusion matrix yesterday, Today I learnt the concept of precision, recall and F1 score performance parameter in binary classification.

### Precision
   It is known as positive predictive value, measures the proportion of true positive prediction out of all positive predictions. It helps to understand the accuracy of positive predictions.

 - **For Example:** Let's take an example of logistic regression model which classifies the email as 'Spam' or 'Not Spam'. Now, if our model predicts a mail as 'Spam' but in real if it not actually 'Spam' then its a blunder, right? So, in this case, we use precision performance parameter to reduce False Positive.
   
**Formula of Precision:** (True Positive)/(True Positive + False Positive)

### Recall
   It measures the proportion of true positive predictions out of all actual positives. It helps to understand how well the model identifies positive instances.

- **For Example:** If a person is suffering from cancer but our model predicts that s/he is not suffering from cancer then thats again a huge blunder. In this case, recall parameter is used to reduce False Negative.

**Formula of Recall:** (True Positive)/(True Positive + False Negative)

### F1 Score
   The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, especially when dealing with imbalanced datasets.

   - **Example:** The prediction of stock market whether it will crash tomorrow or not, we have to use F1 score i.e. both precision and recall as we have to reduce both false positive and false negative. 

   - **Note:** It is one of the kind of F-beta score where both reducing both false positive and fasle negative is important.

# Day 7
Today, I grasped the basic concept of Support Vector Machines(SVM) which is used to solve the problems of both supervised learning i.e. Regression and Classification. In SVM, we need to find the best fitting hyperplane and along with that hyperplane we need to find parallel vectors which is referred as Marginal plane.
    Today, I learned about SVC only. I got know that while classifying the dataset, we have to make sure to make best fitting hyperplane and their supporting vectors. The most importtant thing here is that the distance between the marginal planes should be more so as to reduce the probability of misclassification.

# Day 8
 
 Today, I learnt the concept of Decision Tree in machine learning. It structures decision based on input data, making it suitable for both classification and regression tasks that's why it is called **CART** i.e. Classification and REgression Tree. Its Geometric Intuition is that it converts hyperplane parallel to any of the axes into hyper cuboid in 3D space. 

 ### Entropy
 Entropy is the measure of Disorder or the measure of purity/impurity or measure of uncertainity of random variables.

 - More Knowledge, less Entropy and vice versa
 - Formula:
    H(X) = – Σ (pi * log2 pi)

### Entropy vs Probability
![fig](entropy-in-machine-learning6.webp)
 
 In the system, all the probability is either 0 or 1 then entropy is 0 but if the probability is somewhere in the middle then entropy changes and is no longer 0.

 ### Information Gain

 It is a metric used to train decision trees. Specifically, this metric measures the quality of split.

 - Formula:
   Information Gain = E(Parent)- {Weighted Average}*E(Children)

 ### Gini Impurity
  
  Gini Impurity is a measurement used to build Decision Trees to determine how the features of a dataset should split nodes to form the tree. It is more similary to that of entropy. More precisely, the Gini Impurity of a dataset is a number between 0-0.5, which indicates the likelihood of new, random data being misclassified if it were given a random class label according to the class distribution in the dataset.


# Day 9
Today, I did practical implementation for Decision tree using Post Prunning technique. It is the technique in which tree model is built first and then parameters are used later to ensure balanced tree and good accuracy. The test is performed on iris dataset.

- Below is code snippet:
![photo](DecisionTreeCode_Part1.png)
![photo](DecisionTree_Part1.png)

# Day 10
I did decision tree classification using pre-prunning technique where I gave all possible parameters and used GridSearchCV to find the best params and apply it to get accuracy of 93.33 which is very good and I hope there is less overfiting of the data by the model.

- Below is the code snippet:
![photo](DecisionTree_part2.png)

# Day 11
Today, I did Decision Tree  Regression along with its indepth intuition. Since Decision tree is also called CART algorithm, we can use this for regression problem. For classification, we used various hyperparameters like Entropy, Gini Impurity, Information Gain, etc. Similary to that for regressor, we calculate MSE or MAE which is variance in general.

- formula for mse:
   
   MSE = (1/n) * Σ(actual – forecast)2

And we do variance reduction and find out the decision node. If the variance reduction is higher than we chose them.

- formula for variance reduction is:
   
   Var(Root)- Σwi * Var(child)

# Day 12

Today's topic is **Ensemble Learning** which some way blew out my mind. I got to know about the concept called **Wisdom of the Crowd**. Ensemble learning is based on the same concept. Ensemble learning is a machine learning technique that enhances accuracy and resilience in forecasting by merging predictions from multiple models. It aims to mitigate errors or biases that may exist in individual models by leveraging the collective intelligence of the ensemble. The underlying concept behind ensemble learning is to combine the outputs of diverse models to create a more precise prediction. It works on both:
 
 - Classification: uses majority prediction
 - Regression: uses mean value of all prediction.

#### Implementation
Ensemble learning should have different base models for prediction. It can be achieve in 2 main ways:

- Using different ML models,
- Using same ML models but training them on different datasets


#### Types of Ensemble Learning
- Voting Ensemble
- Bagging also called (**Boostrapped Aggregration**)
- Boosting
- Stacking

### Why Ensemble Learning works?

- For classification

![photo](Example-of-Combining-Decision-Boundaries-Using-an-Ensemble.webp)

- For Regression

![photo](Example-of-Combining-Hyperplanes-Using-an-Ensemble.webp)

# Day 13
Today, I did **Voting Ensemble Classification**. In this, I implemented both the techniques of Ensemble that using different algorithms and using same algo with different dataset. I also got to know the concept of **Soft** voting and **Hard** voting in Voting Ensemble.

- **Hard Voting:**In hard voting (also known as majority voting), every individual classifier votes for a class, and the majority wins. In statistical terms, the predicted target label of the ensemble is the mode of the distribution of individually predicted labels.

- **Soft Voting:** every individual classifier provides a probability value that a specific data point belongs to a particular target class. The predictions are weighted by the classifier's importance and summed up. Then the target label with the greatest sum of weighted probabilities wins the vote.

- Below is code snippet:

![alt text](VotingEnsembleClassifier.png)

# Day 14
Did Ensemble Learning on Voting Ensemble Regressor on dataset from sklearn i.e. Diabetes dataset and did it using both the techniques. I used Linear Regressor, Simple Vector Regresor and Decision Tree Regressor.

- Below is code snippet:
![alt text](VotingRegressor.png)

# Day 15
Today Out of Curiosity, I decided to learn the implementation of Linear Regression from Scratch. I did implement with the help of python code and I will be trying to do others as well in coming future.

- Below is the code Snippet:

![alt text](LR_From_Scratch.png)

![alt text](LR_Testing.png)

# Day 16
  During My learning in machine learning, I started bagging ensemble learning. It is similarly to that of voting ensemble. **Bagging** stands for **BootStrapping and Aggregration**. The key difference here is that bagging is used with decision trees, where it significantly raises the stability of models in improving accuracy and reducing variance, which eliminates the challenge of overfitting.The dataset into various subsets and those subset are feed to that models.  
        I also studied the concept of **Random Forest** which isd  one of the powerful bagging concept.Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.Random decision forests correct for decision trees' habit of overfitting to their training set. More on Random Forest soon!

