# Import necessary libraries
import pickle


# Load model
loaded_model = pickle.load(open("breast_cancer_model.pkl", "rb"))


# Define the features
features = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error",
    "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]

# Collect input for each feature
user_input = {}
for feature in features:
    while True:
        feature = feature.replace("_", " ").title()  # Format feature name for display
        print(f"Feature: {feature}")
        try:
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)  # Convert input to float
            break
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.")
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)
        except Exception as e: 
            print(f"An error occurred: {e}. Please enter a numeric value.")
            value = input(f"Enter value for {feature}: ")
            user_input[feature] = float(value)
            
# Display the collected user input



print("User input collected:", user_input)