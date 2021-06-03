import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd




st.write("""
# OGA BOGA LAND
""")
st.write('test')

st.write('---')
st.write('jk is testing this web')


#df_train = pd.read_csv('../DATA/train_TIN200.csv')
#C:\Users\jkfyr\OneDrive\Documents\NMBU\Tin200\Tin200\DATA
olo = pd.read_csv('C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/prepared_train.csv')
url = 'C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/train_TIN200.csv'
#print(olo)
st.write(olo)
st.write('---')
"""
print(float(olo.CoapplicantIncome.min()))
print(float(olo.CoapplicantIncome.max()))
print(float(olo.CoapplicantIncome.mean()))
"""

def user_value():
    #Loan_ID =
    Gender_Imputed = st.sidebar.slider('Gender', 0, 1)
    Married_Imputed = st.sidebar.slider('Married', 0, 1)
    Dependents_Imputed = st.sidebar.slider('Dependents', 0, 1, 3)
    Education_Imputed = st.sidebar.slider('Education', 0, 1)
    #Self_Employed_Imputed = st.sidebar.slider('Self_Employed', 0, 1)
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', float(olo.ApplicantIncome.min()), float(olo.ApplicantIncome.max()), float(olo.ApplicantIncome.mean()))
    CoapplicantIncome= st.sidebar.slider('CoapplicantIncome', float(olo.CoapplicantIncome.min()), float(olo.CoapplicantIncome.max()), float(olo.CoapplicantIncome.mean()))
    LoanAmount= st.sidebar.slider('LoanAmount', float(olo.LoanAmount.min()), float(olo.LoanAmount.max()), float(olo.LoanAmount.mean()))
    Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term ', float(olo.Loan_Amount_Term .min()), float(olo.Loan_Amount_Term .max()), float(olo.Loan_Amount_Term .mean()))
    Credit_History = st.sidebar.slider('Credit_History', float(olo.Credit_History.min()), float(olo.Credit_History.max()), float(olo.Credit_History.mean()))
    Property_Area_Imputed = st.sidebar.slider('Property_Area', 0, 1, 2)
    #Loan_Status = st.sidebar('Loan_Status', float(olo.Loan_Status.min()), float(olo.Loan_Status.max()), float(olo.Loan_Status.mean()))

    data = { #'Loan_ID' : 0,
            'Gender': Gender_Imputed,
            'Married': Married_Imputed,
            'Dependents': Dependents_Imputed,
            'Education': Education_Imputed,
            #'Self_Employed': Self_Employed_Imputed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area_Imputed,
            #'Loan_Status' :Loan_Status
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_value()
st.write('Input values')
st.table(input_df)
st.write('---')


