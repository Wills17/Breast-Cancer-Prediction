from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and data
model = pickle.load(open("breast_cancer_model.pkl", "rb"))
data = pd.read_csv("Dataset/breast_cancer_data.csv")

# Prepare features
features = data.columns.tolist()
features.remove("Unnamed: 32")
features.remove("diagnosis")


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

@app.route("/")
def home():
    return render_template("home.html", features=features, display_names=display_names)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = {}
    for feature in display_names.items():
        try:
            user_input[feature] = float(request.form.get(feature))
        except:
            user_input[feature] = 0.0

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)

    result = "Malignant" if prediction[0] == 1 else "Benign"
    return render_template("predict.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
