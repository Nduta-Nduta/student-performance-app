🎓 Student Performance Prediction System

End-to-End Machine Learning Pipeline + Deployment

An end-to-end machine learning application that predicts student academic performance using demographic, behavioral, and academic indicators.

This project demonstrates full data science lifecycle execution — from data preprocessing and feature engineering to model evaluation and deployment in a production-ready Streamlit interface.

⸻

🚀 Project Objective

To build a predictive system capable of identifying students at academic risk using structured educational data.

The model can be used by:
	•	Educational institutions
	•	Academic advisors
	•	EdTech platforms
	•	Data-driven intervention teams

⸻

🧠 Technical Implementation

1️⃣ Data Processing
	•	Cleaned and validated structured dataset
	•	Handled missing values and outliers
	•	Encoded categorical variables using appropriate encoding techniques
	•	Feature scaling for model stability
	•	Feature selection based on correlation and importance metrics

⸻

2️⃣ Exploratory Data Analysis (EDA)
	•	Distribution analysis of academic indicators
	•	Correlation heatmaps
	•	Feature importance analysis
	•	Identification of high-impact predictors

⸻

3️⃣ Model Development

Models Tested:
	•	Logistic Regression
	•	Random Forest Classifier
	•	(Optional) Gradient Boosting

Training Approach:
	•	Train/Test split
	•	Cross-validation
	•	Hyperparameter tuning
	•	Overfitting checks

Evaluation Metrics:
	•	Accuracy
	•	Precision / Recall
	•	F1 Score
	•	Confusion Matrix

The final model was selected based on performance stability and generalization ability.

⸻

📊 Model Performance
	•	High predictive accuracy on unseen data
	•	Balanced precision and recall
	•	Strong generalization across performance categories

(Exact metrics available in the notebook under notebooks/eda.ipynb)

⸻

💻 Deployment

The trained model is deployed using Streamlit, providing:
	•	Interactive input fields
	•	Real-time predictions
	•	Clean UI with light/dark mode
	•	Instant inference response
	•	Scalable structure for future cloud deployment

⸻

🛠 Tech Stack
	•	Python
	•	Pandas
	•	NumPy
	•	Scikit-learn
	•	Matplotlib
	•	Streamlit
	•	Joblib
