from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    # Get the user input from the form
    delivery_person_age = float(request.form['delivery_person_age'])
    delivery_person_ratings = float(request.form['delivery_person_ratings'])
    restaurant_latitude = float(request.form['restaurant_latitude'])
    restaurant_longitude = float(request.form['restaurant_longitude'])
    delivery_location_latitude = float(request.form['delivery_location_latitude'])
    delivery_location_longitude = float(request.form['delivery_location_longitude'])
    multiple_deliveries = float(request.form['multiple_deliveries'])
    weather_conditions = request.form['weather_conditions']
    road_traffic_density = request.form['road_traffic_density']
    type_of_order = request.form['type_of_order']
    type_of_vehicle = request.form['type_of_vehicle']
    festival = request.form['festival']

    # Create a new DataFrame with user-provided input values
    new_data = pd.DataFrame({
        'Delivery_person_Age': [delivery_person_age],
        'Delivery_person_Ratings': [delivery_person_ratings],
        'Restaurant_latitude': [restaurant_latitude],
        'Restaurant_longitude': [restaurant_longitude],
        'Delivery_location_latitude': [delivery_location_latitude],
        'Delivery_location_longitude': [delivery_location_longitude],
        'multiple_deliveries': [multiple_deliveries],
        'Weather_conditions_Cloudy': 0,
        'Weather_conditions_Fog': 0,
        'Weather_conditions_Sandstorms': 0,
        'Weather_conditions_Stormy': 0,
        'Weather_conditions_Sunny': 0,
        'Weather_conditions_Windy': 0,
        'Road_traffic_density_High': 0,
        'Road_traffic_density_Jam': 0,
        'Road_traffic_density_Low': 0,
        'Road_traffic_density_Medium': 0,
        'Type_of_order_Buffet': 0,
        'Type_of_order_Drinks': 0,
        'Type_of_order_Meal': 0,
        'Type_of_order_Snack': 0,
        'Type_of_vehicle_electric_scooter': 0,
        'Type_of_vehicle_motorcycle': 0,
        'Type_of_vehicle_scooter': 0,
        'Festival_No': 0,
        'Festival_Yes': 0
    })

    # Map user inputs to binary values for weather conditions
    if weather_conditions in ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy']:
        new_data[f'Weather_conditions_{weather_conditions}'] = 1

    # Map user input to binary values for road traffic density
    if road_traffic_density in ['High', 'Jam', 'Low', 'Medium']:
        new_data[f'Road_traffic_density_{road_traffic_density}'] = 1

    # Map user input to binary values for type of order
    if type_of_order in ['Buffet', 'Drinks', 'Meal', 'Snack']:
        new_data[f'Type_of_order_{type_of_order}'] = 1

    # Map user input to binary values for festival
    if festival in ['No', 'Yes']:
        new_data[f'Festival_{festival}'] = 1
    # Map user input to binary values for festival

    if type_of_vehicle in ['electric_scooter', 'motorcycle', 'scooter']:
        new_data[f'Type_of_vehicle_{type_of_vehicle}'] = 1

    # Scale the features
    new_data_scaled = scaler.transform(new_data)

    # Use the trained model to get the predicted delivery time
    y_pred = model.predict(new_data_scaled)

    # Render the results template with the predicted delivery time
    return render_template('results.html', delivery_time=y_pred.item())

@app.route('/templates/index.html')
def go_home():
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
