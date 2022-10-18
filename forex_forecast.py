# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 08:45:45 2022

@author: Raulin L. Cadet
"""
import pandas
import numpy
import matplotlib
import matplotlib.pyplot
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import matplotlib.dates as mdates
from datetime import datetime

########################################################
###
def forex_request(symbol1,symbol2,frequency):
    import requests
    import json
    import pandas 
    my_api='CJA0RCBPJIL23J4D'
    if frequency=='d':
        url='https://www.alphavantage.co/query?function=FX_DAILY&from_symbol='+symbol1+'&to_symbol='+symbol2+'&outputsize=full'+'&apikey='+my_api
    elif frequency=='w':
        url='https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol='+symbol1+'&to_symbol='+symbol2+'&outputsize=full'+'&apikey='+my_api
    elif frequency=='m':
        url='https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol='+symbol1+'&to_symbol='+symbol2+'&outputsize=full'+'&apikey='+my_api
   
    r = requests.get(url)
    data = r.json()
    dat=[i for i in data.items()]
   
    df=pandas.DataFrame([j for i,j in [i for i in dat[1][1].items()]])
    df['Dates']=sdates=[d for (d,v) in [i for i in dat[1][1].items()]]
    df=df.set_index('Dates');df.columns=['Open','High','Low','Close']
    
    # change data to numeric, since they are strings
    y=[]
    for c in df.columns:
        y.append(pandas.to_numeric(df[c]))
    df2=pandas.DataFrame(y).transpose()
    #df2=df2.reset_index('Dates');df2.columns=['Open','High','Low','Close']
    
    return df2[::-1] # to reverse data, so that most recent data appear at the tail of the data frame
######

#########################################################
###### Function to request FRED time series   ##########
########################################################
def fred_request_series(series,frequency='a',starttime='1776-07-04',endtime='9999-12-31',transform='lin'):
    import requests
    import json
    fred_api='3f3ea2b88220ca8b204bdbb8a5ced854'
    url='https://api.stlouisfed.org/fred/series/observations?series_id='+series+'&output_type=1'+'&frequency='+frequency+'&units='+transform+'&observation_start='+starttime+'&observation_end='+endtime+'&api_key='+fred_api+'&file_type=json'
    r = requests.get(url)
    data = r.json()
    dat=pd.DataFrame([i for i in data.values()][12])
    dat=dat[['date','value']]
    dat.columns=['Dates',series]
    dat=dat.set_index('Dates')
    ###### Since missing values are strings, I convert to float #######
    def to_float(x):
        y=[]
        for i in x:
            try:
                y.append(float(i))
            except:
                y.append(np.nan)
        return y 
    z=[]
    for i in dat.columns:
        z.append(to_float(dat[i]))
        #####
    dat2=pd.DataFrame(z).transpose()   # build a data frame with the lists of float data
    dat2.index=dat.index;dat2.columns=dat.columns
    return dat2
#########################

###############################################
######   Definition of the Series used   ######
###############################################
series_definition=dict({
    # 'DCOILBRENTEU':'Crude Oil Prices - Brent Europe - $ per barrel',
    # 'DCOILWTICO':'Crude Oil Prices - West Texas Intermediate (WTI) - Cushing, Oklahoma - $ per barrel',
    # 'DFII10':'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed - in %',
    # 'DFII30':'Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed',
    'NASDAQCOM': 'NASDAQ current Composite Index',
    'SP500':'S&P 500 current index',
    'BAMLHE00EHYIEY':'ICE BofA Euro High Yield Index Effective Yield - in %'
    })

######## To verify the last date of FRED series  #########
# for i in list(series_definition.keys()):
#     print((fred_request_series(i,frequency='d').index[-1]))
# ###################################################################


####################################################################
#####    Function to merge data frames of series from FRED   #######
####################################################################
def fred_merge(series_list,frequency):
    d=fred_request_series(series_list[0],frequency=frequency,starttime='2005-01-01',transform='pch')
    for i in range(1,len(series_list)):
        d=pd.merge(d,fred_request_series(series_list[i],frequency=frequency,starttime='2005-01-01',transform='pch'),on='Dates' )
    return d


###################################################################
########       Function to create new features            #########
###################################################################


def forex_features_create(x): # x is a data frame of FOREX data
    day=[];difsum=[];sum_ohlc=[];dif_hl=[];dif_oc=[]
    for i in range(0,x.shape[0]):
        day.append(datetime.strptime(x.index[i],'%Y-%m-%d').weekday()+1)
        difsum.append(abs(x.High[i]-x.Low[i])/(x.Open[i]+x.Close[i]))
        sum_ohlc.append(np.mean([x.Open[i]+x.High[i]+x.Low[i]+x.Close[i]]))
        dif_hl.append(x.High[i]-x.Low[i])
        dif_oc.append(x.Open[i]-x.Close[i])
    x['Day']=day
    x['difsum_forex']=difsum
    x['Sum_forex']=sum_ohlc
    x['Dif_HL']=dif_hl
    x['Dif_OC']=dif_oc  
    return x
###################
def fred_features_create(x): # x is data frame of FRED series
    import itertools
    import pandas
    pairs=[i for i in itertools.combinations(iterable=x.columns.tolist(), r=2)]
    ysum=[];yminus=[];yprod=[]
    for p in pairs:
        ysum.append(abs(x[p[0]]+x[p[1]])/x[p[0]])
        yminus.append(abs(x[p[0]]-x[p[1]])/x[p[1]])
        yprod.append(abs(x[p[0]]*x[p[1]])/(x[p[0]]+x[p[1]]))
    df_sum=pandas.DataFrame(ysum).transpose()
    df_minus=pandas.DataFrame(yminus).transpose()
    df_prod=pandas.DataFrame(yprod).transpose()
    df_sum.columns=[i[0]+'_sum_'+i[1] for i in pairs]
    df_minus.columns=[i[0]+'_minus_'+i[1] for i in pairs]
    df_prod.columns=[i[0]+'_prod_'+i[1] for i in pairs]
    
    df=pd.merge(pd.merge(df_sum,df_minus, on='Dates'), df_prod,on='Dates' )
    df=pd.merge(x,df,on='Dates')
    return df

###########################################################
#####  Function to select features based on Variance ######
###########################################################
def features_select(x): # x is the data frame or array of variables
    from sklearn.feature_selection import VarianceThreshold
    selected = VarianceThreshold(threshold=0.001)
    X=selected.fit_transform(x) # array of data of selected variables
    cols = selected.get_support(indices=True)   # index of selected variables
    cols_names=x.columns[cols]                  # names of selected variables
    return X,cols_names
    
############################################################
#### Function to normalize data to range 0,1
###########################################################
def normalized_df(x):
    def normalized(z):
        return [(i-min(z))/(max(z)-min(z))+.005 for i in z]
    h=[]
    for j in x.columns:
        if j=='Day':        # if the feature j is Day, its values are kept without normalization
            h.append(x[j])
        else:
            h.append(normalized(x[j]))
    d=pd.DataFrame(h).transpose()
    d.index=x.index;d.columns=x.columns
    return d

###########################################################
#####    RETREIVING DATA AND FEATURES ENGENEERING   #######
###########################################################
# dat_forex=forex_request('USD', 'EUR', frequency='d') # EUR for 1 USD
# df=pd.merge(dat_forex,fred_merge(list(series_definition.keys())),on='Dates')
# features_create(df)   # new data frame created with this line of code
############################################################


##########################################################
####    Function to realize machine learning        #####
##########################################################

def forex_learning(symbol2,symbol1='USD',endogeneous='Close',frequency='d',lag=2):
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import metrics
    
    dat_forex=forex_features_create(forex_request(symbol1, symbol2, frequency=frequency)) # EUR for 1 USD
    df=pd.merge(dat_forex,fred_features_create(fred_merge(list(series_definition.keys()),frequency=frequency)),on='Dates')
 
    # forex_features_create(df)   # new data frame created with this line of code
    # df=fred_features_create(df)
    ###########
    # dat_forex=forex_request(symbol1, symbol2, frequency=frequency) 
    # # dat_fred=fred_merge(list(series_definition.keys())
    # # forex_features_create(dat_forex)
    # # fred_features_create(dat_fred)
    # df=pd.merge(dat_forex,fred_merge(list(series_definition.keys()),on='Dates')
    #########
    df2=df.shift(lag).iloc[lag:,:]
    df2=pd.merge(df[endogeneous],df2.iloc[:,5:],on='Dates')
    # Defined label and features
    # y=df[endogeneous].iloc[:-lag]
    y=df[endogeneous][lag:]
    # X=df2[df2.columns[df2.columns!=endogeneous]]
    X=df2
    # normalize the features
    Xf=normalized_df(X)
    
    # select featurest
    features_selected=features_select(Xf)
    X=features_selected[0]          # features selected
    features_columns=features_selected[1]    # list of features names
    Xf=Xf[features_columns] # data frame of selected features to be used for forecasting new y
    # replace missing value by mean value
    imput_missing = SimpleImputer(missing_values=np.nan, strategy='mean')
    X=imput_missing.fit_transform(X)
    Xf=imput_missing.fit_transform(Xf) 
    Xf=pd.DataFrame(Xf)
    Xf.columns=features_columns;Xf.index=df2.index
    # splitting data to train and test groups 
    X_train, X_test, y_train, y_test= train_test_split(X, y,train_size=0.25,random_state=0)  
    lr=np.linspace(0.01,0.5,num=10)
    rmse=[];mae=[]
    for r in lr:
        mod=GradientBoostingRegressor(learning_rate=r)
        mod.fit(X_train,y_train)
        y_pred=mod.predict(X_test)
        rmse.append(metrics.mean_squared_error(y_test, y_pred,squared=False)) # root mean square error (mean squqre error if squared=True)
        mae.append(metrics.mean_absolute_error(y_test,y_pred))
    return dict(zip(lr,rmse)),X_train,y_train,Xf
    


#################################################################################
## Function to forecast future values of the features, then those of the label ##
#################################################################################
def forex_forecast(model_res,n_forecast,lag=2): # n_forecast is the number of forecasting period; model_res: result of forex_learning
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from datetime import datetime, date, timedelta
    from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
    
    def future_dates(x,n_forecast): # returns weekday
        ''' 
        the function returns weekday in this format '%Y-%m-%d'.
        '''
        last_date=datetime.strptime(x[3].index[len(x[3])-1], '%Y-%m-%d') # ok
        fdate=[];
        # fdate.append(datetime.strptime(x[3].index[len(x[3])-1], '%Y-%m-%d')) # last date of available data
        fdate1=[]
        for k in range(1,n_forecast+1): # k is the number of days to add to last date
            # last_date=datetime.strptime(fdate[len(fdate)-1], '%Y-%m-%d') 
            da1=datetime.strftime(last_date+timedelta(days=1),'%Y-%m-%d')
            da2=(datetime.strptime(da1,'%Y-%m-%d')).weekday()
            if (da2==5 or da2==6)==True:
                if (da2==5)==True:
                    fdate.append(datetime.strftime(last_date+timedelta(days=3),'%Y-%m-%d')) # to have date in indicated format; +2 means add 2 days when the date is a saturday
                    fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to remove to replace 0 value by 1 for forecasting exchange rate using day value as features   
                if (da2==6)==True:
                    fdate.append(datetime.strftime(last_date+timedelta(days=2),'%Y-%m-%d')) # to have date in indicated format; +1 means add 1 days when the date is a sunday
                    fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to remove to replace 0 value by 1 for forecasting exchange rate using day value as features   
            else:
                fdate.append(datetime.strftime(last_date+timedelta(days=1),'%Y-%m-%d')) # to have date in indicated format; +2 means add 2 days when the date is a saturday
                fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to replace 0 value by 1 for forecasting exchange rate using day value as features     
            last_date=datetime.strptime(fdate[len(fdate)-1], '%Y-%m-%d')  
        return fdate1
    ####
    def future_dates0(x,n_forecast): # 
        ''' 
        returns weekday as integer
        n_forecast: number of days to forecast data
        '''
        from datetime import datetime, date, timedelta
        last_date=datetime.strptime(x[3].index[len(x[3])-1], '%Y-%m-%d') # ok
        fdate=[];
        # fdate.append(datetime.strptime(x[3].index[len(x[3])-1], '%Y-%m-%d')) # last date of available data
        fdate1=[]
        for k in range(1,n_forecast+1): # k is the number of days to add to last date
            # last_date=datetime.strptime(fdate[len(fdate)-1], '%Y-%m-%d') 
            da1=datetime.strftime(last_date+timedelta(days=1),'%Y-%m-%d')
            da2=(datetime.strptime(da1,'%Y-%m-%d')).weekday()
            if (da2==5 or da2==6)==True:
                if (da2==5)==True:
                    fdate.append(datetime.strftime(last_date+timedelta(days=3),'%Y-%m-%d')) # to have date in indicated format; +2 means add 2 days when the date is a saturday
                    fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to remove to replace 0 value by 1 for forecasting exchange rate using day value as features   
                if (da2==6)==True:
                    fdate.append(datetime.strftime(last_date+timedelta(days=2),'%Y-%m-%d')) # to have date in indicated format; +1 means add 1 days when the date is a sunday
                    fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to remove to replace 0 value by 1 for forecasting exchange rate using day value as features   
            else:
                fdate.append(datetime.strftime(last_date+timedelta(days=1),'%Y-%m-%d')) # to have date in indicated format; +2 means add 2 days when the date is a saturday
                fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to replace 0 value by 1 for forecasting exchange rate using day value as features     
            last_date=datetime.strptime(fdate[len(fdate)-1], '%Y-%m-%d')  
        return fdate
    
        # Forecast values of a list for n periods
    def fX_forecast(x,n_forecast): # x a list to forecast for n_forecast periods
        from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore") # to remove warning from the results of the regression
        #########   
        model_exp1=ExponentialSmoothing(x,
            seasonal_periods=5,
            trend="add", seasonal="add",damped_trend=True,use_boxcox=True,
            initialization_method="estimated").fit()#.forecast(10).rename(r"$\alpha=0.2$") 
        y=model_exp1.forecast(n_forecast)#.rename(r"$\alpha=0.2$") 
        #  # if the forecast values are nan, they are replaced by the last value of the variable
        z=[]
        for f in y: # replace np.nan by the last value of the variable
            if np.isnan(f)==True:
                z.append(x[len(x)-1])
            else:
                z.append(f)    
        return z
        # return y
    
    ################
    # find position of the variable Day if it is selected as feature
    # In this case, its future value should not be forecast, they should be future weekdays
    def position_day(x):
        pres_day1=['Day'==i for i in x[3].columns.tolist()]
        if True in pres_day1: # if 'Day' is one of the selected features
            pos_day=[i for i in range(0,len(pres_day1)) if pres_day1[i]==True][0]
        else:               # if Day is not among selected features
            pos_day=False 
        return pos_day
       ####
    
    if position_day(model_res)==False:  # if 'Day' is not one of the selected features
        yy=[fX_forecast(pd.DataFrame(model_res[3]).iloc[:,i], n_forecast) for i in range(0,len(model_res[3].columns.tolist())) ] # list of forecasted values of the features
    elif position_day(model_res)!=False: # if Day is one of selected features
        yy=[]
        for s in range(0,model_res[3].shape[1]):
            if s==position_day(model_res):                      # if it is the feature 'Day'
                yy.append(future_dates0(model_res, n_forecast))
            else:                                               # if it is another feature 
                yy.append(fX_forecast(pd.DataFrame(model_res[3]).iloc[:,s],n_forecast))
   
    z=pd.DataFrame(yy).transpose()                  # data frame of forecasted features
    z.index=future_dates0(model_res, n_forecast)
    z.columns=model_res[3].columns                  # columns names are those of the features found in model_res[3]
    z=pd.concat([model_res[3].iloc[-lag:,:],z])     # concat forecasted features with last data available for these features
    z=z.shift(lag)                                  # shift data to use lag values of features as it is the case when estimating the model
    z=z.dropna()
    
    ##### Forecast of label for n_forecast period  #####
    r_minRMSE=[i for i,j in model_res[0].items() if j==min(model_res[0].values())][0]
    model1=GradientBoostingRegressor(learning_rate=r_minRMSE)
    model1.fit(model_res[1],model_res[2])
    ypred=model1.predict(z)
    dk=pd.DataFrame(index=z.index.tolist(),data=ypred.tolist())
    dk.columns=['Forecast']
    return dk
################


############################################################
#### FUNCTION TO AUTOMATE THE FORECAST OF EXCHANGE RATE ####
############################################################

def forex_automated_forecast(symbol2,symbol1,endogeneous='Close',frequency='d',lag=2,n_forecast=5):
    model=forex_learning(symbol2=symbol2,symbol1=symbol1,endogeneous=endogeneous,frequency=frequency,lag=lag)
    
    return forex_forecast(model_res=model,n_forecast=n_forecast,lag=lag)
    
##############################

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#########################################################
#@@@@@@      BUILDING STREAMLIT DASHBOARD         @@@@@@#
#########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
n_forecast=11
st.set_page_config(layout="wide")


def fig1():
    import pandas
    st.set_option('deprecation.showPyplotGlobalUse', False)
    dou=pandas.concat([fd.Close,forecastX],axis=1)
    dou.columns=['Historical values','Forecasted values']
    dou.tail(90).plot()
    ###################

    

########################################
import streamlit.components.v1 as components
new_title = '<p style="font-family:sans-serif; color:#36719C; font-size: 60px;">Forecasted Exchange Rate</p>'
st.markdown(new_title, unsafe_allow_html=True)
author = '<p style="font-family:sans-serif; color:#A06357; font-size: 25px;">Raulin L. Cadet</p>'

col1,col2,col3=st.columns([1,1,2])
col1.markdown(author,unsafe_allow_html=True)
# col2.markdown("[![Title](‘https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg’)]('https://www.linkedin.com/in/raulincadet/')")
col2.markdown("['LinkedIn_logo.png']('https://www.linkedin.com/in/raulincadet/')")

st.markdown("---")


# col1,col2,col3=st.columns([1,1,2])
currencies_list=['USD','EUR','JPY', 'GBP', 'AUD','CAD','CHF','CNY','HKD','NZD']
currency1 = col1.selectbox("Select a currency ",sorted(currencies_list))
currency2 = col2.selectbox("Select the other currency ",sorted(currencies_list,reverse=True))

forecastX=forex_automated_forecast(symbol1=currency1,symbol2=currency2,n_forecast=n_forecast)
forecastX=pandas.DataFrame(data=forecastX.values[1:],index=forecastX.index[1:])
fd=forex_request(currency1,currency2,'d')    

montant = col1.number_input('Enter the amount of '+currency1+' to convert to '+currency2,1)
result=col2.number_input('Result',value=montant*fd.tail(1).Close.tolist()[0])
col1.markdown("---")
col2.markdown("---")

col3.markdown('#### Forecasted exchange rate: '+currency2+'/'+currency1)
# col3.pyplot(fig1(currency1,currency2))
col3.pyplot(fig1())
df_forecastX=forecastX#.iloc[-n_forecast:,:]
df_fd=fd.tail(10)

col1.markdown('#### Exchange rate: '+currency2+'/'+currency1)
col1.dataframe(df_fd)
col2.markdown('#### Forecasted: '+currency2+'/'+currency1)
col2.dataframe(df_forecastX)

