# app.py
import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import joblib

# Load data and model
df = pd.read_csv('data/heart.csv')
model = joblib.load('model/model.pkl')

# Initialize app
app = dash.Dash(__name__)
app.title = 'Heart Disease Prediction Dashboard'

# Layout
app.layout = html.Div([
    html.H1("Heart Disease Predictor", style={"textAlign": "center"}),

    html.Div([
        html.Label('Age'),
        dcc.Input(id='age', type='number', min=1, max=120, step=1, value=50),

        html.Label('Sex (1 = Male, 0 = Female)'),
        dcc.Input(id='sex', type='number', min=0, max=1, value=1),

        html.Label('Chest Pain Type (0-3)'),
        dcc.Input(id='cp', type='number', min=0, max=3, value=0),

        html.Label('Resting Blood Pressure'),
        dcc.Input(id='trestbps', type='number', value=120),

        html.Label('Cholesterol'),
        dcc.Input(id='chol', type='number', value=200),

        html.Label('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)'),
        dcc.Input(id='fbs', type='number', min=0, max=1, value=0),

        html.Label('Resting ECG Results (0-2)'),
        dcc.Input(id='restecg', type='number', min=0, max=2, value=1),

        html.Label('Max Heart Rate Achieved'),
        dcc.Input(id='thalach', type='number', value=150),

        html.Label('Exercise Induced Angina (1 = Yes, 0 = No)'),
        dcc.Input(id='exang', type='number', min=0, max=1, value=0),

        html.Br(),
        html.Button('Predict', id='predict-btn', n_clicks=0),
        html.Div(id='prediction-output', style={"marginTop": "20px", "fontWeight": "bold"})
    ], style={"width": "50%", "margin": "auto"}),

    html.Hr(),

    html.Div([
        html.H3("Data Distribution"),
        dcc.Graph(id='chol-dist', figure=px.histogram(df, x='chol', nbins=40, title='Cholesterol Distribution'))
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [
        Input('age', 'value'),
        Input('sex', 'value'),
        Input('cp', 'value'),
        Input('trestbps', 'value'),
        Input('chol', 'value'),
        Input('fbs', 'value'),
        Input('restecg', 'value'),
        Input('thalach', 'value'),
        Input('exang', 'value'),
    ]
)
def predict(n_clicks, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang):
    if n_clicks > 0:
        X = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]],
                         columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang'])
        y_pred = model.predict(X)[0]
        return f"Prediction: {'Heart Disease Detected' if y_pred == 1 else 'No Heart Disease'}"
    return ""

if __name__ == '__main__':
    app.run(debug=True)


