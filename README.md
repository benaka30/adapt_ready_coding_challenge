# adapt_ready_coding_challenge
This repository contains the code and snapshots of results for the problem statement provided- "Electric power distribution involves supplying electricity to various areas, but predicting 
the demand for specific areas is challenging due to factors such as weekdays, holidays, 
seasons, weather, and temperatures. Inaccurate predictions can lead to equipment 
damage or unnecessary energy waste. 
Oil temperature is a critical indicator of a transformer's condition. Accurately forecasting 
the oil temperature can help manage the load on transformers, prevent damage, and 
optimize energy distribution. 
Your challenge is to build a machine learning model to forecast the oil temperature for the 
next 24 hours, predicting values at 1-hour intervals. Additionally, explore the dataset to 
gain insights and evaluate the extreme load capacity."

*************************************************************************************************
About the code-
1)Objective:
We are forecasting the oil temperature for the next 24 hours, predicting values at 1-hour intervals using a Random Forest model.
The model uses past values (lag features) to predict the future temperature.

2)We import essential libraries:
numpy and pandas are used for numerical operations and data handling.
DecisionTreeRegressor: A single decision tree used as a base learner in the custom Random Forest.
mean_squared_error: To evaluate the performance of the model by comparing the actual and predicted values.
train_test_split: Splits the dataset into training and testing sets.

3. Custom Random Forest Class
   __init__: Initializes the Random Forest with parameters:
n_estimators: Number of decision trees in the forest.
max_depth: Maximum depth of each tree.
min_samples_split: Minimum samples required to split an internal node.
n_features: Number of features to consider when looking for the best split.
self.trees: Empty list to hold the individual decision trees.

Bootstrapping: Random sampling with replacement
_bootstrap_sample: Creates random samples (with replacement) of the dataset to train each tree on different data samples.
The result is a "bootstrapped" sample of data (X_sample and y_sample) used to train an individual decision tree.

Fitting the Random Forest:
fit: For each tree in the forest (self.n_estimators), it:
Creates a bootstrapped sample from the training data.
Initializes a decision tree with the given hyperparameters (max_depth, min_samples_split, max_features).
Fits the tree to the bootstrapped data.
Adds the trained tree to the self.trees list.

Loading the Data
The dataset is read using pandas

Model Training
RandomForest: Initializes a custom Random Forest model with 100 trees, a maximum depth of 10, a minimum of 2 samples required to split a node, and 5 features considered at each split.
fit: Trains the model on the training data (X_train, y_train)

The path of the dataset has to be provided properly based on the path on the particular system.
the required libraries must be installed before execution of the code.
Vs code ide was used to execute the code .
