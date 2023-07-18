# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('data.csv')

# Data preprocessing
# Drop any missing values
data = data.dropna()

# Split the dataset into features (X) and target variable (y)
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Make predictions on new data
new_data = pd.DataFrame({'feature1': [value1], 'feature2': [value2], 'feature3': [value3]})
prediction = model.predict(new_data)
print(f"Prediction: {prediction}")
