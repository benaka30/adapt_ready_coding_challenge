import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Custom implementation of a Random Forest
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


# Load the dataset
data = pd.read_csv('C:/Users/benak/Downloads/TempPredict_Assignment/test.csv')

# Use these feature columns
features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']  # Correct feature names from dataset
target = 'OT'  # Oil temperature (target)

# Prepare the feature matrix (X) and target variable (y)
X = data[features].values
y = data[target].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the custom Random Forest model
rf_model = RandomForest(n_estimators=100, max_depth=10, min_samples_split=2, n_features=5)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Forecast oil temperature for the next 24 hours at 1-hour intervals
def forecast_next_24_hours(last_known_data, model, n_steps=24):
    forecasted_temps = []
    
    # Loop through each hour and predict the temperature
    for step in range(n_steps):
        # Use the last known data to predict the next hour's oil temperature
        last_known_features = last_known_data[-1].copy()  # Copy last row
        
        # Predict the oil temperature for the next hour
        next_temp = model.predict(np.array([last_known_features]))
        
        # Append the forecasted temperature
        forecasted_temps.append(next_temp[0])
        
        # Simulate adding the new predicted temperature back into the feature matrix
        next_row = last_known_features  # This can be adjusted based on how you handle input features
        next_row[0] = next_temp  # Update the first feature (e.g., oil temp) with the predicted value
        
        # Append the updated feature set for the next prediction
        last_known_data = np.vstack([last_known_data, next_row])

    return forecasted_temps

# Use the last known data from the training set for forecasting
last_known_data = X_train[-24:]  # Take the last 24 hours as initial data

# Forecast the oil temperature for the next 24 hours
forecasted_temps = forecast_next_24_hours(last_known_data, rf_model)
print("Forecasted Oil Temperatures for the next 24 hours:", forecasted_temps)
