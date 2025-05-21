from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load trained model, scaler, and feature list
with open("model.pkl", "rb") as file:
    model_info = pickle.load(file)

model = model_info["model"]
scaler = model_info["scaler"]
feature_names = model_info["features"]  # These are the renamed, scaled features

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from form
        process_temp = float(request.form['Process_temperature_K'])
        speed = float(request.form['Rotational_speed_rpm'])
        torque = float(request.form['Torque_Nm'])
        tool_wear = float(request.form['Tool_wear_min'])

        input_data = pd.DataFrame([[process_temp, speed, torque, tool_wear]],
    columns=["Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])

        input_scaled = scaler.transform(input_data)


        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "Failure Expected!" if prediction == 1 else "No Failure Expected"

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction="")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)