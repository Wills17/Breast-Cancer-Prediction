# Import necessary libraries
import pickle
import pandas as pd


# Load model
Loaded_model = pickle.load(open("breast_cancer_model.pkl", "rb"))
print("Model loaded successfully.")


# Load the dataset to get feature names
data = pd.read_csv("Dataset/breast_cancer_data.csv")
print("Columns in the dataset:")

features = data.columns.tolist()  # Use all columns as features
 # Remove uneeded columns
features.remove("Unnamed: 32") 
features.remove("diagnosis")
#features.remove("id") 
print(features)

display_names = {
    'id': 'Customer ID',
    'radius_mean': 'Mean Radius',
    'texture_mean': 'Mean Texture',
    'perimeter_mean': 'Mean Perimeter',
    'area_mean': 'Mean Area',
    'smoothness_mean': 'Mean Smoothness',
    'compactness_mean': 'Mean Compactness',
    'concavity_mean': 'Mean Concavity',
    'concave points_mean': 'Mean Concave Points',
    'symmetry_mean': 'Mean Symmetry',
    'fractal_dimension_mean': 'Mean Fractal Dimension',
    'radius_se': 'SE Radius',
    'texture_se': 'SE Texture',
    'perimeter_se': 'SE Perimeter',
    'area_se': 'SE Area',
    'smoothness_se': 'SE Smoothness',
    'compactness_se': 'SE Compactness',
    'concavity_se': 'SE Concavity',
    'concave points_se': 'SE Concave Points',
    'symmetry_se': 'SE Symmetry',
    'fractal_dimension_se': 'SE Fractal Dimension',
    'radius_worst': 'Worst Radius',
    'texture_worst': 'Worst Texture',
    'perimeter_worst': 'Worst Perimeter',
    'area_worst': 'Worst Area',
    'smoothness_worst': 'Worst Smoothness',
    'compactness_worst': 'Worst Compactness',
    'concavity_worst': 'Worst Concavity',
    'concave points_worst': 'Worst Concave Points',
    'symmetry_worst': 'Worst Symmetry',
    'fractal_dimension_worst': 'Worst Fractal Dimension'
}



user_input = {}
for feature, display_name in display_names.items():
    while True:
        print(f"\nFeature: {display_name}")
        try:
            value = input(f"Enter value for {display_name}: ")
            user_input[feature] = float(value)  # Convert input to float
            break
        except ValueError:
            print(f"Invalid input for {display_name}. Please enter a numeric value.\n")
            value = input(f"Enter value for {display_name}: ")
            user_input[feature] = float(value)
        except Exception as e: 
            print(f"An error occurred: {e}. Please enter a numeric value.\n")
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)
            
            
# Display the collected user input
print("User input collected:", user_input)

# Make prediction
user_input = pd.DataFrame([user_input])  # Convert user_input to a DataFrame
prediction = Loaded_model.predict(user_input)

# Display the prediction result
if prediction[0] == 1:
    print("The model predicts: Malignant")
else:
    print("The model predicts: Benign")
    