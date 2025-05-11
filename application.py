
from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np

app = Flask(__name__)

with open("breast_cancer_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = []
        for feature in request.form.values():
            try:
                features.append(float(feature))
            except ValueError:
                return render_template('predict.html', error="Please enter valid numeric values")
        
        features = np.array([features])
        
        prediction = model.predict(features)
        
        result = "Malignant" if prediction == 1 else "Benign"
        
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return render_template('predict.html', error=f"An error occurred: {str(e)}")

@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'No prediction available')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
