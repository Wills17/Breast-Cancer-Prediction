# Import necessary libraries
import pickle
import pandas as pd


# Load model
Loaded_model = pickle.load(open("breast_cancer_model.pkl", "rb"))

# Display the model summary
print("Model loaded successfully.")
print("Model summary:")
print(Loaded_model)


# Load the dataset to get feature names
data = pd.read_csv("breast_cancer_data.csv")  # Replace with the actual dataset path
print("Columns in the dataset:")
print(data.columns.tolist())

features = data.columns.tolist()  # Use all columns as features

# Collect input for each feature
user_input = {}
for feature in features:
    while True:
        feature = feature.replace("_", " ").title()  # Format feature name for display
        print(f"\nFeature: {feature}")
        try:
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)  # Convert input to float
            break
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.\n")
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)
        except Exception as e: 
            print(f"An error occurred: {e}. Please enter a numeric value.\n")
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)
            
# Display the collected user input
print("User input collected:", user_input)

# Prepare the input data for prediction
input_data = [user_input[feature.replace(" ", "_").lower()] for feature in features]


# Make prediction
prediction = Loaded_model.predict([input_data])

# Display the prediction result
if prediction[0] == 1:
    print("The model predicts: Malignant")
else:
    print("The model predicts: Benign")