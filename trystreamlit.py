# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:18:02 2022

@author: Diaraye
"""
import streamlit as st
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.dates as mdates
# col1,col2 = st.beta_columns([3,1])
#################################3333
dates=pandas.date_range('2020-01-01', periods=200, freq='w');dates
dfou=pandas.DataFrame(np.random.rand(200,2))
dfou.index=dates
dfou.columns=['X1','X2']
##
fig,ax=plt.subplots()
ax.plot(dfou.X1,label='Historical')
ax.plot(dfou.X2,label='Forecasted')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.plot(forecastX.index,forecastX.Forecast)

# plt.title('Exchange rate: '+currency1+'/'+currency2)
ax.xlabel('Dates')
# plt.ylabel(currency1+'/'+currency2)
plt.xticks(rotation=30)
plt.show()

# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%d'))

###############################333
st.title('Forecasted Exchange Rate')
col1,col2=st.columns(2)

fig=plt.figure()
plt.plot([1,2,3,4,5],[1,3,6,4,9])
plt.fill_between(x=[1,2,3,4,5],y1=[1,3,6,4,9], color='lightblue')
plt.show()
# forex_automated_forecast('GBP',n_forecast=10).plot()
# fig.savefig(fig)
# img = Image.open("fig.png")
col1.pyplot(fig)
col2.pyplot(fig)

montant = col1.number_input("Enter your weight (in kgs)",1)
# st.text('Selected: {}'.format(montant))
# result=col2.text("Amount{}: ".format(montant*2))
result=col2.number_input('Result',value=montant*2,max_value=0)