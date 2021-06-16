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
from eli5 import explain_weights
from sklearn.model_selection import train_test_split

seed = 1337

st.title("Loan checker")
st.sidebar.title('Enter your information here:')


#region IMAGE
#TODO: IMAGE WIll not load on web

#img = r'C:\Users\jkfyr\OneDrive\Documents\NMBU\Tin200\Tin200\Code\monster.png'
#img = imread('monster.png')
#st.image(img, use_column_width=True, width=5)
#endregion

#st.write("[FIRE!!!]()")

# TODO: fix URLS stremlit will not run short rout
# ===============================
st.write('---')
"""
Hi, potential customer! \n
If you want a loan you can try out this program to
check whether your loan will be approved or not. Just plug in your data on the left hand side
and let us do the rest. Remember, we will only return estimates and you 
will have to consult our bank to actually get a loan. 
If our program decides you won't get a loan you can check
at the bottom to see what you may want to improve or change in order to get a loan
"""
#components.html("<p style='color:red;'> :O COLOR")

#url = 'C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/DATA/prepared_train.csv'
df = pd.read_csv(r'C:/Users/jkfyr/OneDrive/Documents/NMBU/Tin200/Tin200/final_draft.csv')


def user_value():
    #Loan_ID = st.sidebar.s
    Gender_Imputed = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Married_Imputed = st.sidebar.selectbox('Married', [ 'No', 'Yes'])
    Dependents_Imputed = st.sidebar.slider('Dependents', 0, 1, 3)
    Education_Imputed = st.sidebar.selectbox('Education', ['No', 'Yes'])
    Self_Employed_Imputed = st.sidebar.selectbox('Self_Employed', ['No', 'Yes'])
    ApplicantIncome = st.sidebar.slider('ApplicantIncome', float(df.ApplicantIncome.min()), float(df.ApplicantIncome.max()), float(df.ApplicantIncome.mean()))
    CoapplicantIncome= st.sidebar.slider('CoapplicantIncome', float(df.CoapplicantIncome.min()), float(df.CoapplicantIncome.max()), float(df.CoapplicantIncome.mean()))
    LoanAmount= st.sidebar.slider('LoanAmount', float(df.LoanAmount.min()), float(df.LoanAmount.max()), float(df.LoanAmount.mean()))
    Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term ', float(df.Loan_Amount_Term .min()), float(df.Loan_Amount_Term .max()), float(df.Loan_Amount_Term .mean()))
    Credit_History = st.sidebar.slider('Credit_History', 0, 1)
    Property_Area_Imputed = st.sidebar.slider('Property_Area', 0, 1, 2)
    #Loan_Status = st.sidebar('Loan_Status', float(olo.Loan_Status.min()), float(olo.Loan_Status.max()), float(olo.Loan_Status.mean()))

    data = { #'Loan_ID' : 0,
            'Gender': Gender_Imputed,
            'Married': Married_Imputed,
            'Dependents': Dependents_Imputed,
            'Education': Education_Imputed,
            'Self_Employed': Self_Employed_Imputed,
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
st.write('---')
display_df = input_df.transpose()
display_df = display_df.rename(columns={0: 'Your info'})
st.subheader('Your information:')

st.write(display_df)


string_val = ['Gender', 'Married', 'Education', 'Self_Employed']
for col in string_val:

    if input_df.iloc[0][col] == 'Male':
        input_df.loc[0, [col]] = 0

    elif input_df.iloc[0][col] == 'No':
        input_df.loc[0, [col]] = 0
    else:
        input_df.loc[0, [col]] = 1


st.write('---')

#st.subheader('Tuned DataFrame')
#profile = ProfileReport(olo)
#st_profile_report(profile)

#region placeholder_regresser
#X = olo.drop('Loan_Status')




X = df.drop('Loan_Status', axis=1)

#print(Y.columns)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)


forest = RandomForestClassifier()
forest.fit(X_train, y_train)

forest.fit(X, y)
pred = forest.predict(input_df)
#st.write(pred)

#xgb = XGBClassifier()
#xgb.fit(X, y)
#pred_xg = xgb.predict(input_df)

if pred[0] == 0:
    disp_pred = 'Not approved'

else:
    disp_pred = 'Approved'

st.header('Your loan is: {}'.format(disp_pred))


import eli5
from eli5.sklearn import PermutationImportance

if disp_pred == 'Not approved':
    droplist = ['Gender_Imputed', 'Married_Imputed', 'Property_Area_Imputed', 'Dependents_Imputed',
                'Self_Employed_Imputed']

    """
    Here is your information that affected our decision the most.
    You may want to look into these to see if you can 
    make any changes in order to increase your chances of getting a loan.
    Good luck!

    """

    perm = PermutationImportance(forest).fit(X, y)
    weights_forest = eli5.show_weights(perm, feature_names=list(X.columns))
    weights_forest = pd.read_html(weights_forest.data)[0]
    improvements = weights_forest[~weights_forest['Feature'].isin(droplist)]
    improvements = improvements.reset_index(drop=True)

    st.write(improvements['Feature'])
else:
    """
    Congratulations! We think you may be able to get a loan! Reach out to us
    so we can double check the application and set it all up.
    Looking forward to seeing you soon!
    """
# endregion
