from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

print(f"Template folder: {os.path.abspath('templates')}")
print(f"Static folder: {os.path.abspath('static')}")
print(f"Template folder exists: {os.path.isdir('templates')}")
print(f"Static folder exists: {os.path.isdir('static')}")

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    print("Model and scaler loaded successfully")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get input values from form
        try:
            hours_studied = float(request.form["hours_studied"])
            attendance = float(request.form["attendance"])
            previous_grade = float(request.form["previous_grade"])

            # Scale the input data
            data = np.array([[hours_studied, attendance, previous_grade]])
            data_scaled = scaler.transform(data)
            
            # Predict final grade
            prediction = model.predict(data_scaled)[0]
            prediction = round(prediction, 2)
            print(f"Prediction: {prediction}")
        except Exception as e:
            prediction = f"Error: {str(e)}"
            print(f"Error in prediction: {e}")

    print(f"Rendering template with prediction={prediction}")
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True) 
