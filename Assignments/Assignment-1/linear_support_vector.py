import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
glass_data = pd.read_csv(".\data\glass.csv")

# Split the dataset into features (X) and labels (y)
X = glass_data.drop('Type', axis=1)
y = glass_data['Type']

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear SVM classifier
classifier = SVC(kernel='linear')

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