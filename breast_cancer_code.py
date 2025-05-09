#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
data = pd.read_csv('Dataset/breast_cancer_data.csv')

# Display the first 10 rows of the dataset 
"""print(data.head(10))"""

# Display the shape
"""print(data.shape)"""

# Check for missing values
"""print(data.info())"""
"""print(data.isnull().sum())"""

"""print(data.describe())"""

data = data.drop(columns=['Unnamed: 32'])
# Display the first 10 rows of the dataset after dropping the column
print(data.head(10))
"""print("Shape of data:", data.shape)"""

"""print(data["diagnosis"].value_counts())"""

# Plot the distribution of the diagonis
sns.countplot(data["diagnosis"], label="Count")
plt.title("Count of Malignant and Benign Tumors")
plt.xticks(rotation=90)
plt.legend()
"""plt.show()"""
plt.close()

# Convert the diagnosis column to numerical values with one-hot encoding
Encode = LabelEncoder()
data["diagnosis"] = Encode.fit_transform(data["diagnosis"])

print(data["diagnosis"].value_counts())


print(data.corr())
# Plot correlation matrix
plt.figure(figsize=(30, 30))
plt.title("Correlation Matrix")
sns.heatmap(data.corr(), fmt=".2f", cmap="coolwarm")
"""plt.show()"""
plt.close()

# Create Pairplot based on diagnosis
sns.pairplot(data.iloc[:, 1:5], hue="diagnosis") 
"""plt.show()"""
plt.close()

# target and features
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.fit_transform(X_test)

# Train the logistic regression model
models = [LogisticRegression(random_state=30), LinearRegression(), 
         RandomForestClassifier(random_state=30)]
for model in models:
    model.fit(X_train, y_train)

    #Evaluate training data model
    Score = model.score(X_train, y_train)
    print("Accuracy report on Training set for {}: {:.2f}%".format(model, Score*100))


# Use Random Forest Classifier
LR = LogisticRegression(random_state=30)
LR_model = LR.fit(X_train, y_train)

# Make predictions
y_pred = LR_model.predict(X_test)

# Make evaluations
LR_model_score = accuracy_score(y_test, y_pred)
print("\nAccuracy score using Random Forest: {:.2f}%".format(LR_model_score*100))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))