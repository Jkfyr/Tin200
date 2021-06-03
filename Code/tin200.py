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
    Gender_Imputed = st.sidebar.slider('Gender', float(olo.Gender_Imputed.min()), float(olo.Gender_Imputed.max()), float(olo.Gender_Imputed.mean()))
    Married_Imputed = st.sidebar.slider('Married', 0, 1)
    Dependents_Imputed = st.sidebar.slider('Dependents', float(olo.Dependents_Imputed.min()), float(olo.Dependents_Imputed.max()), float(olo.Dependents_Imputed.mean()))
    Education_Imputed = st.sidebar.slider('Education', float(olo.Education_Imputed.min()), float(olo.Education_Imputed.max()), float(olo.Education_Imputed.mean()))
    #Self_Employed_Imputed = st.sidebar.slider('Self_Employed', float(olo.Self_Employed_Imputed.min()), float(olo.Self_Employed.max()), float(olo.Self_Employed_Imputed.mean()))
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', float(olo.ApplicantIncome.min()), float(olo.ApplicantIncome.max()), float(olo.ApplicantIncome.mean()))
    CoapplicantIncome= st.sidebar.slider('CoapplicantIncome', float(olo.CoapplicantIncome.min()), float(olo.CoapplicantIncome.max()), float(olo.CoapplicantIncome.mean()))
    LoanAmount= st.sidebar.slider('LoanAmount', float(olo.LoanAmount.min()), float(olo.LoanAmount.max()), float(olo.LoanAmount.mean()))
    Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term ', float(olo.Loan_Amount_Term .min()), float(olo.Loan_Amount_Term .max()), float(olo.Loan_Amount_Term .mean()))
    Credit_History = st.sidebar.slider('Credit_History', float(olo.Credit_History.min()), float(olo.Credit_History.max()), float(olo.Credit_History.mean()))
    Property_Area_Imputed = st.sidebar.slider('Property_Area', float(olo.Property_Area_Imputed.min()), float(olo.Property_Area_Imputed.max()), float(olo.Property_Area_Imputed.mean()))
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


