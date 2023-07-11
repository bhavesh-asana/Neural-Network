import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# a) Import the dataset
salary_data = pd.read_csv(".\data\Salary_Data.csv")

# Split the data into features (X) and target variable (y)
X = salary_data['YearsExperience'].values.reshape(-1, 1)
y = salary_data['Salary'].values

# b) Split the data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# c) Train and predict the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

# d) Calculate the mean squared error
mse = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error:", mse)

# e) Visualize the train and test data using a scatter plot
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, y_train_pred, color='green', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression')
plt.legend()
plt.show()
