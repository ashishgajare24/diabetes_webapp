# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:35:41 2022

@author: ashis
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav','rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    

def main():
    
    # giving a title
    st.title('Diabetes Prediction ')
    st.caption('Made with ❤ by Ashish Gajare', unsafe_allow_html=False)
    
    # getting the input data from user
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.number_input('Glucose Level')
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value')
    
    with col2:
        Insulin = st.number_input('Insulin Level')
    
    with col3:
        BMI = st.number_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.number_input('Age of the Person')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Predict'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        st.success(diagnosis)
        
if __name__ == '__main__':
    main()
