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
olo = pd.read_csv('C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/train_TIN200.csv')
url = 'C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/train_TIN200.csv'
#print(olo)
st.write(olo)
st.write('---')
print(float(olo.CoapplicantIncome.min()))
print(float(olo.CoapplicantIncome.max()))
print(float(olo.CoapplicantIncome.mean()))


def user_value():
    # Loan_ID =
    # Gender=
    # Married=
    # Dependents=
    # Education=
    # Self_Employed=
    #ApplicantIncome = st.sidebar('ApplicantIncome', float(olo.ApplicantIncome.min()), float(olo.ApplicantIncome.max()),float(olo.ApplicantIncome.mean()))
    CoapplicantIncome= st.sidebar('CoapplicantIncome', float(olo.CoapplicantIncome.min()), float(olo.CoapplicantIncome.max()),float(olo.CoapplicantIncome.mean()))
    #LoanAmount= st.sidebar('LoanAmount', float(olo.LoanAmount.min()), float(olo.LoanAmount.min()),float(olo.LoanAmount.mean()))
    #Loan_Amount_Term = st.sidebar('Loan_Amount_Term ', float(olo.Loan_Amount_Term .min()), float(olo.Loan_Amount_Term .min()),float(olo.Loan_Amount_Term .mean()))
    # Credit_History=
    # Property_Area=
    # Loan_Status=
    data= {
    #'Loan_ID' : 0,
    #'Gender' : 0,
    #'Married' : 0,
    #'Dependents' : 0,
    #'Education' : 0,
    #'Self_Employed' : 0,
    #'ApplicantIncome' : ApplicantIncome,
    'CoapplicantIncome' :CoapplicantIncome,
    #'LoanAmount': LoanAmount,
    #'Loan_Amount_Term' : LoanAmount,
    #'Credit_History' : 0,
    #'Property_Area' :0,
    #'Loan_Status' :0
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_value()
st.write('Input values')
st.table(input_df)
st.write('---')


