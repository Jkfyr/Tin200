import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('../DATA/train_TIN200.csv')

print(df_train)


st.write("""
# Doge FOMO Calculator
Use the option below to set your parameters for:
- Date You Wish You Would Have Bought $Doge
- USD Amount You Wish You Would Have Invested
""")
st.write('---')


#st.write("[coindesk]()")
