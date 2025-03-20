import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

## Loead the Trained Model

model = tf.keras.models.load_model('model.keras')

#Load the encoders and scalers

with open('label_encoder_gender.pkl', 'rb') as file:  # here we open the file with the intention to read
    label_encoder_gender = pickle.load(file) # load the pickle file into the variable
    
with open('label_encoder_geography.pkl', 'rb') as file:  # here we open the file with the intention to read
    label_encoder_geography = pickle.load(file) # load the pickle file into the variable
    
with open('scaler.pkl', 'rb') as file:  # here we open the file with the intention to read
    scaler = pickle.load(file) # load the pickle file into the variable
    

## Streamlit App

st.title('Customer Chrun Prediction')

geography = st.selectbox('Geopgraphy', label_encoder_geography.categories_[0]) 
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode 'Gender'
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode 'Geography'

geo_encoded = label_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geography.get_feature_names_out(['Geography']))

# Concatenate the input data and the one-hot encoded 'Geography'
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)   # Here reset_index(drop=True) is used to reset the index of the input_data dataframe

#Scale the input data
input_data_scaled = scaler.transform(input_data)


#Predict the churn

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba >= 0.5:
    st.write('The customer is likely to churn')
    st.write(f'Prediction Probability: {prediction_proba}')
else:
    st.write('The customer is not likely to churn')
    
    