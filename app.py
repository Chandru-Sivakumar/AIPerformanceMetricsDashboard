from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import os
import joblib

app = Flask(__name__)

# Directory to store generated plots
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load trained models with error handling
try:
    revenue_model = joblib.load("models/revenue_model.pkl")
    dash_model = joblib.load("models/dashboard_model.pkl")
    dash_encoder = joblib.load("models/dashboard_encoder.pkl")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
    revenue_model, dash_model, dash_encoder = None, None, None

def generate_matplotlib_plot():
    """Generate and save a simple matplotlib bar chart."""
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Actual', 'Predicted'], y=[100, np.random.randint(80, 120)], palette='Blues')
    plt.ylabel('Revenue')
    plot_path = os.path.join(PLOT_DIR, 'matplotlib_plot.png')
    plt.savefig(plot_path, format='png')  # Ensure correct format
    plt.close()
    return plot_path

def generate_plotly_chart(cpu_usage):
    """Generate a Plotly gauge chart and return it as a JSON object."""
    simulated_cpu_usage = np.random.randint(10, 100)  # Simulated CPU usage value
    if not cpu_usage:
        cpu_usage = simulated_cpu_usage  # Use simulated CPU usage if no input provided
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cpu_usage,  
        title={'text': "CPU Usage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "red"}
            ],
        }
    ))

    return gauge_chart.to_json()  # Convert the Plotly figure to JSON


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not revenue_model or not dash_model or not dash_encoder:
        return jsonify({"error": "Model files are missing. Please check your setup."}), 500

    try:
        # Define the feature keys
        feature_keys = ["cpu_usage", "memory_usage", "disk_usage", "response_time", 
                        "error_rate", "active_users", "new_users", "session_duration", 
                        "revenue", "customer_retention_rate", "churn_rate", 
                        "model_accuracy", "precision", "recall", "f1_score", 
                        "inference_time", "anomaly_detected"]

        # Convert form data to numpy array
        inputs = np.array([[float(request.form[key]) for key in feature_keys]])

        # Extract CPU Usage for Plotly Gauge Chart
        cpu_usage = float(request.form["cpu_usage"])

        # Make predictions
        revenue_prediction = revenue_model.predict(inputs)[0]
        dashboard_element_pred = dash_model.predict(inputs.reshape(1, -1))
        dashboard_element = dash_encoder.inverse_transform(dashboard_element_pred)[0]

        # Generate visualizations
        matplotlib_plot = generate_matplotlib_plot()
        plotly_chart_dict = generate_plotly_chart(cpu_usage)  # Now returns a dictionary

        return jsonify({
            'revenue_prediction': float(revenue_prediction),
            'dashboard_element': dashboard_element,
            'matplotlib_plot': '/' + matplotlib_plot,
            'plotly_data': plotly_chart_dict  # Send dictionary instead of JSON string
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
