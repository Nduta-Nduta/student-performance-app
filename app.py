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
    error = None
    if request.method == "POST":
        try:
            hours_studied = float(request.form.get("hours_studied", 0))
            attendance = float(request.form.get("attendance", 0))
            previous_grade = float(request.form.get("previous_grade", 0))
            sleep_hours = float(request.form.get("sleep_hours", 0))
            study_group = float(request.form.get("study_group", 0))
            assignment_completion = float(request.form.get("assignment_completion", 0))
            test_prep_days = float(request.form.get("test_prep_days", 0))

            # Validate inputs
            if not (0 <= attendance <= 100):
                error = "Attendance must be between 0 and 100"
            elif not (0 <= hours_studied <= 168):
                error = "Hours studied must be between 0 and 168"
            elif not (0 <= previous_grade <= 100):
                error = "Previous grade must be between 0 and 100"
            elif not (0 <= sleep_hours <= 12):
                error = "Sleep hours must be between 0 and 12"
            elif not (0 <= study_group <= 10):
                error = "Study group participation must be between 0 and 10"
            elif not (0 <= assignment_completion <= 100):
                error = "Assignment completion must be between 0 and 100"
            elif not (0 <= test_prep_days <= 14):
                error = "Test prep days must be between 0 and 14"
            else:
                # Scale and predict
                data = np.array([[hours_studied, attendance, previous_grade, sleep_hours, study_group, assignment_completion, test_prep_days]])
                data_scaled = scaler.transform(data)
                prediction = model.predict(data_scaled)[0]
                prediction = round(float(prediction), 2)
                print(f"Prediction: {prediction}")
        except ValueError:
            error = "Please enter valid numeric values"
        except Exception as e:
            error = f"Prediction error: {str(e)}"
            print(f"Exception: {e}")

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True) 
