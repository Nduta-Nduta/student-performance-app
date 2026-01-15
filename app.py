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
            practice_hours = float(request.form["practice_hours"])
            lessons_taken = float(request.form["lessons_taken"])
            theory_score = float(request.form["theory_score"])
            mock_test_score = float(request.form["mock_test_score"])
            missed_lessons = float(request.form["missed_lessons"])

            # Scale the input data
            data = np.array([[practice_hours, lessons_taken, theory_score, mock_test_score, missed_lessons]])
            data_scaled = scaler.transform(data)
            
            # Predict pass probability
            prediction_proba = model.predict_proba(data_scaled)[0][1]
            prediction = round(prediction_proba * 100, 2)
        except (ValueError, KeyError) as e:
            prediction = f"Invalid input. Please enter numeric values. Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True) 
