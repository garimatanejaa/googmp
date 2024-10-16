from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
#scaler = joblib.load('scaler.pkl')

# Define thresholds for traffic volume to categorize congestion levels
threshold_high = 1000  # Example value for high congestion
threshold_low = 300     # Example value for low congestion

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        is_holiday = int(request.form['is_holiday'])
        temperature = float(request.form['temperature'])
        weekday = int(request.form['weekday'])
        hour = int(request.form['hour'])
        month_day = int(request.form['month_day'])
        year = int(request.form['year'])
        month = int(request.form['month'])

        # Prepare input data
        input_data = np.array([[is_holiday, temperature, weekday, hour, month_day, year, month]])
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Determine congestion level
        if prediction[0] > threshold_high:
            congestion_level = 'High'
        elif prediction[0] < threshold_low:
            congestion_level = 'Low'
        else:
            congestion_level = 'Medium'

        return jsonify({'traffic_volume': prediction[0], 'congestion_level': congestion_level})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
