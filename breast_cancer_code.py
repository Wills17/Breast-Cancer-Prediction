#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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
#plt.show()
plt.close()

# Convert the diagnosis column to numerical values with one-hot encoding
Encode = LabelEncoder()
data["diagnosis"] = Encode.fit_transform(data["diagnosis"])

print(data["diagnosis"].value_counts())


print(data.corr())
plt.figure(figsize=(30, 30))
# Plot the correlation matrix
plt.title("Correlation Matrix")
sns.heatmap(data.corr(), fmt=".2f", cmap="coolwarm")
plt.show()
plt.close()

