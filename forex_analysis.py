# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:50:51 2022

@author: Raulin L. Cadet
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button
import sklearn
import json
import seaborn as sb
import pytrends

##########################
#my_api='21RLS0795C533255'
##########################

def forex_request(symbol1,symbol2,frequency):
    my_api='CJA0RCBPJIL23J4D'
  
    url= 'https://www.alphavantage.co/query?function='+frequency+'&from_symbol='+symbol1+'&to_symbol='+symbol2+'&outputsize=full'+'&apikey='+my_api
    r = requests.get(url)
    data = r.json()
    dat=[i for i in data.items()]
   
    df=pd.DataFrame([j for i,j in [i for i in dat[1][1].items()]])
    df['Dates']=sdates=[d for (d,v) in [i for i in dat[1][1].items()]]
    df=df.set_index('Dates');df.columns=['Open','High','Low','Close']
    
    # change data to numeric, since they are strings
    y=[]
    for c in df.columns:
        y.append(pd.to_numeric(df[c]))
    df2=pd.DataFrame(y).transpose()
    #df2=df2.reset_index('Dates');df2.columns=['Open','High','Low','Close']
    
    return df2[::-1] # to reverse data, so that most recent data appear at the tail of the data frame
######
dat00=forex_request('EUR', 'USD', frequency='FX_DAILY')
##########################################################

def currency_list():
    return pd.read_csv('physical_currency_list.csv')
##################################################################

def forest_percent_change(x):
    '''
    Parameters
    ----------
    x : TYPE
        a data frame with numeric data in each column
    Returns
    -------
    y : TYPE
        a data frame with the growth rate in each column
    '''
    def changePerc(x):  
        y=[np.nan];i=0
        while i<len(x)-1:
            y.append(((x[i+1]-x[i])/x[i])*100)
            i=i+1
        return y
    z=[]
    for c in x.columns:
        h=x[c].tolist()
        ch=changePerc(h)    # calculate the growth rate
        z.append(ch)        # happend list of growth rate to z, to create a data frame with it
    df=pd.DataFrame(z).transpose()
    df.columns=x.columns
    df.index=list(x.index)
    return df
###
dat01=forest_percent_change(dat00)
##################################################

def forex_plot(x):
    '''
    Parameters
    ----------
    x : TYPE
        a data frame of numeric data
    col : TYPE
        list of type of price for which to show the plot. They are:
            'Open', 'High', 'Low', 'Close'

    Returns
    -------
    None.

    '''
    fig,axs=plt.subplots(2,2)
    axs[0,0].plot(x.index,x.Open)
    axs[0, 0].set_title('Open')
    axs[0,1].plot(x.index,x.High)
    axs[0, 1].set_title('High')
    axs[1,0].plot(x.index,x.Close)
    axs[1, 0].set_title('Close')
    axs[1,1].plot(x.index,x.Low)
    axs[1, 1].set_title('Low')
    plt.show()
####
forex_plot(dat00)

##################################################    
#####       OTHER VARIABLES         #######
###########################################
'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'


def other_variables(frequency):
    my_api='CJA0RCBPJIL23J4D'
    def otherVar_request(URL):
        # my_api='CJA0RCBPJIL23J4D'
      
        url= URL#'https://www.alphavantage.co/query?function='+frequency+'&from_symbol='+symbol1+'&to_symbol='+symbol2+'&outputsize=full'+'&apikey='+my_api
        r = requests.get(url)
        data = r.json()
        dat=[i for i in data.items()]
       
        df=pd.DataFrame([j for i,j in [i for i in dat[1][1].items()]])
        df['Dates']=sdates=[d for (d,v) in [i for i in dat[1][1].items()]]
        df=df.set_index('Dates');df.columns=['Open','High','Low','Close','Volume']
        
        # change data to numeric, since they are strings
        y=[]
        for c in df.columns:
            y.append(pd.to_numeric(df[c]))
        df2=pd.DataFrame(y).transpose()
        #df2=df2.reset_index('Dates');df2.columns=['Open','High','Low','Close']
        
        return df2[::-1] # to reverse data, so that most recent data appear at the tail of the data frame
   #---------
    def federal_fund(URL):
       r = requests.get(URL)
       data = r.json()
       dat=pd.DataFrame([i for i in data.items()][3][1])
       dat.columns=['Dates','FederalFund']
       dat=dat.set_index('Dates')
       return dat[::-1]
       
   #--------
    if frequency=='daily':
       dnasdaq=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NQ'+'&outputsize=full'+'&apikey='+my_api)
       deuronex=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=ENX.PA'+'&outputsize=full'+'&apikey='+my_api)
       dfedfund=federal_fund('https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=weekly&apikey='+my_api)
   
    if frequency=='weekly':
       dnasdaq=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=NQ'+'&outputsize=full'+'&apikey='+my_api)
       deuronex=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=ENX.PA'+'&outputsize=full'+'&apikey='+my_api)
       dfedfund=federal_fund('https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=weekly&apikey='+my_api)
   
    if frequency=='monthly':
        dnasdaq=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=NQ'+'&outputsize=full'+'&apikey='+my_api)
        deuronex=otherVar_request('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=ENX.PA'+'&outputsize=full'+'&apikey='+my_api)
        dfedfund=federal_fund('https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey='+my_api)
    return dnasdaq,deuronex,dfedfund
 #########################3

dou=other_variables('monthly')
