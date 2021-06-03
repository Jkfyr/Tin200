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

print(float(olo.ApplicantIncome.max()))

def user_value():
    # Loan_ID =
    # Gender=
    # Married=
    # Dependents=
    # Education=
    # Self_Employed=
    ApplicantIncome = st.sidebar('ApplicantIncome', float(olo.ApplicantIncome.min()), float(olo.ApplicantIncome.min()),float(olo.ApplicantIncome.mean()))
    CoapplicantIncome= st.sidebar('ApplicantIncome', float(olo.CoapplicantIncome.min()), float(olo.CoapplicantIncome.min()),float(olo.CoapplicantIncome.mean()))
    # LoanAmount=
    # Loan_Amount_Term=
    # Credit_History=
    # Property_Area=
    # Loan_Status=
    data= {
    'Loan_ID' : 0,
    'Gender' : 0,
    'Married' : 0,
    'Dependents' : 0,
    'Education' : 0,
    'Self_Employed' : 0,
    'ApplicantIncome' :0,
    'CoapplicantIncome' :0,
    'LoanAmount':0,
    'Loan_Amount_Term' : 0,
    'Credit_History' : 0,
    'Property_Area' :0,
    'Loan_Status' :0

    }


