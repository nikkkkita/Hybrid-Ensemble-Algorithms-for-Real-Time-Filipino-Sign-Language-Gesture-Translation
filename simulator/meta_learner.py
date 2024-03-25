from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import csv

# Load dataset from CSV file
def load_dataset_from_csv(filename):
    dataset = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append([float(val) for val in row])
    return np.array(dataset)

# Load dataset
dataset = load_dataset_from_csv('dataset.csv')

# Split dataset into features (X) and target probabilities (y)
X = dataset[:, :-1]  # All columns except the last one
y = dataset[:, -1]   # Last column (probability)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression meta-learner
meta_learner = LinearRegression()
meta_learner.fit(X_train, y_train)

# Make predictions on the test set
y_pred = meta_learner.predict(X_test)

# Evaluate the meta-learner performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
