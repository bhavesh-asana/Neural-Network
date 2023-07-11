import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load dataset using pandas
dataframe = pd.read_csv(".\data\glass.csv")
# print(dataframe.head(3))

# Split dataset into features(X) and labels (y)
X = dataframe.drop('Type', axis=1)
y = dataframe['Type']

# Split dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Naïve Bayes classifier
classifier = GaussianNB()

# Train the classifier
classifier.fit(X_train, y_train)

# Evaluate the model on the test part
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

# Predict the labels for the test set
y_pred = classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:")
print(report)