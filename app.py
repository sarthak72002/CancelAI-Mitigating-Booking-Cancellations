import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open('final_model_XGBoost.pkl', 'rb') as file:
    model = pickle.load(file)

# Define your prediction function here
def prediction(days_booking, booking_type, special_requests, price_per_room, adults_per_room,
               weekend_nights, parking_included, week_nights, day_of_arrival, month_of_arrival, weekday_of_arrival):
    
    # Create a numpy array of the input features (ensure the input order matches the model's expectation)
    input_data = np.array([[days_booking, booking_type, special_requests, price_per_room, adults_per_room,
                            weekend_nights, parking_included, week_nights, day_of_arrival, month_of_arrival, weekday_of_arrival]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # You might want to return a more user-friendly result (e.g., 'Cancelled' or 'Not Cancelled')
    if prediction[0] == 1:
        return f'This booking is more likely to canceled: chances = {round(prediction*100,2)}%'
    else:
        return f'This booking is less likely to get canceled: chances = {round(prediction*100,2)}%'

# Streamlit App
st.title('INN Group of Hotels')
st.write('This application will forecast the cancellation of bookings')

# Input Fields
days_booking = st.number_input('HOW MANY PRIOR DAYS BOOKING WAS MADE', min_value=0)
booking_type = st.selectbox('HOW THE BOOKING WAS MADE', options=[('Online', 1), ('Offline', 0)])[1]
special_requests = st.selectbox('HOW MANY SPECIAL REQUESTS MADE', options=[0, 1, 2, 3, 4, 5])
price_per_room = st.number_input('WHAT IS THE PRICE PER ROOM OFFERED', min_value=0.0)
adults_per_room = st.selectbox('HOW MANY ADULTS PER ROOM', options=[1, 2, 3, 4])
weekend_nights = st.number_input('HOW MANY WEEKEND NIGHTS IN THE STAY', min_value=0)
parking_included = st.selectbox('DOES BOOKING INCLUDE PARKING FACILITY', options=[('YES', 1), ('NO', 0)])[1]
week_nights = st.number_input('HOW MANY WEEK NIGHTS IN STAY', min_value=0)
day_of_arrival = st.slider('WHAT IS DAY OF ARRIVAL', min_value=1, max_value=31, step=1)
month_of_arrival = st.slider('WHAT IS MONTH OF ARRIVAL', min_value=1, max_value=12, step=1)
weekday_of_arrival = st.selectbox('WHAT IS THE WEEKDAY OF ARRIVAL', 
                                  options=[('Mon', 0), ('Tue', 1), ('Wed', 2), ('Thu', 3), 
                                           ('Fri', 4), ('Sat', 5), ('Sun', 6)])[1]

# Prediction Button
if st.button('Predict'):
    result = prediction(days_booking, booking_type, special_requests, price_per_room, adults_per_room,
                        weekend_nights, parking_included, week_nights, day_of_arrival, month_of_arrival, weekday_of_arrival)
    st.text('Prediction: ' + result)

