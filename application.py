# Import librairies
from flask import Flask, render_template, request, url_for, redirect
import pickle
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

with open("Dataset/breast_cancer_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


# Configure upload folder and allowed extensions
UPLOAD_FOLDER = "advanced/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form
        features = []
        for feature in request.form.values():
            try:
                features.append(float(feature))
            except ValueError:
                return render_template("predict.html", error="Please enter valid numeric values")
        
        features = np.array([features])
        
        prediction = model.predict(features)
        
        result = "Malignant" if prediction > 0.5 else "Benign"
        
        return render_template("result.html", prediction=result)
    
    except Exception as e:
        return render_template("predict.html", error=f"An error occurred: {str(e)}")

@app.route("/predict-with-images", methods=["GET", "POST"])
def predict_with_images():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("predict_with_images.html", error="No file part")
        
        file = request.files["file"]
        
        # If the user does not select a file
        if file.filename == "":
            return render_template("predict_with_images.html", error="No image uploaded")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Here, you can add code to process the image and make predictions
            # For now, we"ll just return a success message
            return render_template("result.html", prediction="Image uploaded successfully!")
    
    return render_template("predict_with_images.html")

@app.route("/result")
def result():
    prediction = request.args.get("prediction", "No prediction available")
    return render_template("result.html", prediction=prediction)

@app.after_request
def cleanup_uploaded_files(response):
    try:
        for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Error cleaning up uploaded files: {str(e)}\nPlease contact administrator.")
    return response

if __name__ == "__main__":
    app.run(debug=True)
