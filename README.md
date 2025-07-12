# 🫀 Heart Disease Prediction Dashboard

A web-based interactive dashboard built using [Dash](https://dash.plotly.com/) and [scikit-learn](https://scikit-learn.org/) to predict the presence of heart disease based on patient health data.

---

## 📊 Features

- Predicts heart disease using a trained **RandomForestClassifier**
- Clean, responsive UI with input fields
- Interactive visualizations using Plotly
- Uses sample UCI heart dataset
- Model can be retrained with `train_model.py`

---

## 🚀 Demo

![App Screenshot](https://user-images.githubusercontent.com/salarmastoi110/preview.png)

> ⚠️ Replace this with a real screenshot or GIF of the dashboard in action.

---

## 🧠 Machine Learning

- **Model**: Random Forest
- **Dataset**: UCI Heart Disease dataset
- **Features Used**:
  - age
  - sex
  - cp (chest pain type)
  - trestbps (resting blood pressure)
  - chol (cholesterol)
  - fbs (fasting blood sugar)
  - restecg (ECG results)
  - thalach (max heart rate)
  - exang (exercise-induced angina)

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/heart-disease-prediction-dashboard.git
cd heart-disease-prediction-dashboard

# (Optional) create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
