import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('data.csv')

# Split data into train and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Prepare data for training
train_X = train_data.drop(['target'], axis=1).values
train_y = train_data['target'].values

# Initialize linear regression model
model = LinearRegression()

# Train model
model.fit(train_X, train_y)

# Predict using trained model
test_X = test_data.drop(['target'], axis=1).values
test_y = test_data['target'].values
pred_y = model.predict(test_X)

# Evaluate model using mean squared error
mse = mean_squared_error(test_y, pred_y)
print('Mean squared error: ', mse)
