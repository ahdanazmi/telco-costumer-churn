import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

# load preprocessing

with open('scaler.pkl', 'rb') as file_1:
  scaler = joblib.load(file_1) 

with open('encoder.pkl', 'rb') as file_2:
  encoder = joblib.load(file_2) 

model = tf.keras.models.load_model('model.h5')

# widget input
st.title("Predict Churn User")

with st.form(key= 'form_parameter'):
    gender = st.selectbox("Gender", ["male", "female"])
    SeniorCitizen = st.selectbox("Senior Citizen (1 for yes, 0 for no)", [1, 0])
    Partner = st.selectbox("Does the customer has a partner? ", ["Yes","No"])
    Dependents = st.selectbox("Dependents customer", ["Yes","No"])
    tenure = st.number_input("tenure")
    PhoneService = st.selectbox("Does the customer has phone service?", ["Yes","No"])
    MultipleLines = st.selectbox("MultipleLines", ["Yes","No","No phone service"])
    InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes","No","No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["Yes","No","No internet service"])
    TechSupport = st.selectbox("TechSupport", ["Yes","No","No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["Yes","No","No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes","No"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("MonthlyCharges")
    TotalCharges = st.number_input("TotalCharges")
    st.markdown('---')


    submitted = st.form_submit_button('Predict')



# input to dataframe
data_inf = {'gender': gender,
         'SeniorCitizen': SeniorCitizen,
         'Partner' : Partner,
         'Dependents' :Dependents,
         'tenure' : tenure,
         'PhoneService' : PhoneService,
         'MultipleLines' : MultipleLines,
         'InternetService' : InternetService,
         'OnlineSecurity' : OnlineSecurity,
         'OnlineBackup' : OnlineBackup,
         'DeviceProtection' : DeviceProtection,
         'TechSupport' : TechSupport,
         'StreamingTV' : StreamingTV,
         'StreamingMovies' : StreamingMovies,
         'Contract' : Contract,
         'PaperlessBilling' : PaperlessBilling,
         'PaymentMethod' : PaymentMethod,
         'MonthlyCharges' : MonthlyCharges,
         'TotalCharges' : TotalCharges
         }
data_inf = pd.DataFrame([data_inf])

num = ["tenure", "MonthlyCharges", "TotalCharges"]
cat = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
# preprocessing
if submitted:
        # make numerik and categorical data
        data_inf_num = data_inf[num]
        data_inf_cat = data_inf[cat]

        # Feature Scaling
        data_inf_num_scaled = scaler.transform(data_inf_num)
        
        # Feature Encoding
        data_inf_cat_scaled =encoder.fit_transform(data_inf_cat)

        # concate numerical and categorical columns
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_scaled], axis=1)


        # melakukan prediksi
        y_pred_inf = model.predict(data_inf_final)
        y_pred_inf_str = np.where(y_pred_inf >= 0.5, 1, 0)

        if y_pred_inf == 1:
          output= 'Churn'
        else:
          output= 'Not churn'
    

        st.write('## Prediction = ', output )
        