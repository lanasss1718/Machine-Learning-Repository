# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read in data and split into training and testing sets
data = pd.read_csv('data.csv')
X = data.drop('target_variable', axis=1)
y = data['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train score: ", train_score)
print("Test score: ", test_score)
