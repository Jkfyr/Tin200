import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from eli5.xgboost import explain_weights_xgboost
from eli5 import format_as_dataframe
st.title("OGA BOGA LAND")
st.sidebar.title('Enter your data here:')


#region IMAGE
#TODO: IMAGE WIll not load on web

#img = 'C:\Users\jkfyr\OneDrive\Documents\NMBU\Tin200\Tin200\Code\monster.png'
#img = imread('monster.png')
#st.sidebar.image(img, use_column_width=True, width=5)
#endregion

st.write("[FIRE!!!]()")

# TODO: fix URLS stremlit will not run short rout
# ===============================
st.write('---')
st.write('jk is testing this web')
components.html("<p style='color:red;'> :O COLOR")
#df_train = pd.read_csv('../DATA/prepared_train.csv')
#print((df_train))
#url = 'C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/prepared_train.csv'
olo = pd.read_csv('C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/prepared_train.csv')
st.write(olo)
st.write('---')


def user_value():
    #Loan_ID = st.sidebar.s
    Gender_Imputed = st.sidebar.selectbox('Gender', ['Male', 'Fmale'])
    Married_Imputed = st.sidebar.selectbox('Married', ['Yes', 'No'])
    Dependents_Imputed = st.sidebar.slider('Dependents', 0, 1, 3)
    Education_Imputed = st.sidebar.selectbox('Education', ['Yes', 'No'])
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
st.write(input_df)

lb = LabelEncoder()
string_val = ['Gender', 'Married', 'Education']
for col in string_val:
    input_df[col] = lb.fit_transform(input_df[col])


st.write('---')

#st.subheader('Tuned DataFrame')
#profile = ProfileReport(olo)
#st_profile_report(profile)

#region placeholder_regresser
#X = olo.drop('Loan_Status')
X = olo.drop(['Unnamed: 0', 'Loan_Status', 'Loan_ID', 'Self_Employed_Imputed'], axis=1)

Y = olo['Loan_Status']
lb = LabelEncoder()
Y = lb.fit_transform(Y)

#forest = RandomForestClassifier()
#forest.fit(X, Y)
#pred =forest.predict(input_df)
xgb = XGBClassifier()
xgb.fit(X, Y)
pred = xgb.predict(input_df)

if pred[0] == 0:
    pred = 'Not approved'
else:
    pred = 'Approved'

st.header('Your loan is: {}'.format(pred))

params = explain_weights_xgboost(xgb, top=3)
params = format_as_dataframe(params)

st.write('your params:')
st.write(params)
st.write('---')
# endregion