# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 08:45:45 2022

@author: Raulin L. Cadet
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import sklearn
########################################################
###
def forex_request(symbol1,symbol2,frequency):
    import requests
    import json
    import pandas as pd
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
     

#########################################################
### Function to transform data frame to n lag value #####
#########################################################
def lag_transform(x,n):
    import numpy as np
    import pandas as pd
    def lag_values(s,n):
        for i in range(0,n):
            s.append(np.nan)
        return s[n:-n]
    y=[]
    for i in x.columns:
        y.append(lag_values(x[i].tolist(),n))
    df=pd.DataFrame(y) .transpose()
    df.index=x.index[:len(x.index)-n]
    df.columns=x.columns
    return df   

do=pd.merge(pd.merge(fred_request_series('GDP').tail(50),fred_request_series('GDPCA').tail(50),on='Dates'),fred_request_series('DCOILWTICO').tail(50),on='Dates')
# do1=lag_transform(do,n=3)
# do.corr(method='kendall')

###############################################
######   Definition of the Series used   ######
###############################################
series_definition=dict({
    'DCOILBRENTEU':'Crude Oil Prices - Brent Europe - $ per barrel',
    'DCOILWTICO':'Crude Oil Prices - West Texas Intermediate (WTI) - Cushing, Oklahoma - $ per barrel',
    'DFII10':'Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed - in %',
    'DFII30':'Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity, Quoted on an Investment Basis, Inflation-Indexed',
    'NASDAQCOM': 'NASDAQ current Composite Index',
    'SP500':'S&P 500 current index',
    'BAMLHE00EHYIEY':'ICE BofA Euro High Yield Index Effective Yield - in %'
    })

####################################################################
#####    Function to merge data frames of series from FRED   #######
####################################################################
def fred_merge(series_list):
    d=fred_request_series(series_list[0],frequency='d',starttime='2005-01-01',transform='pch')
    for i in range(1,len(series_list)):
        d=pd.merge(d,fred_request_series(series_list[i],frequency='d',starttime='2005-01-01',transform='pch'),on='Dates' )
    return d


###################################################################
########       Function to create new features            #########
###################################################################
from datetime import datetime
def features_create(x): # x is a data frame
    day=[];difsum=[];sum_ohlc=[];dif_hl=[];dif_oc=[]
    for i in range(0,x.shape[0]):
        day.append(datetime.strptime(x.index[i],'%Y-%m-%d').weekday())
        difsum.append((df.High[i]-df.Low[i])/(df.Open[i]+df.Close[i]))
        sum_ohlc.append(np.mean([df.Open[i]+df.High[i]+df.Low[i]+df.Close[i]]))
        dif_hl.append(df.High[i]-df.Low[i])
        dif_oc.append(df.Open[i]-df.Close[i])
    x['Day']=day
    x['difsum_forex']=difsum
    x['Sum_forex']=sum_ohlc
    x['Dif_HL']=dif_hl
    x['Dif_OC']=dif_oc  
    return x


###########################################################
#####  Function to select features based on Variance ######
###########################################################
def features_select(x): # x is the data frame or array of variables
    from sklearn.feature_selection import VarianceThreshold
    selected = VarianceThreshold(threshold=0.01)
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
        h.append(normalized(x[j]))
    d=pd.DataFrame(h).transpose()
    d.index=x.index;d.columns=x.columns
    return d

###########################################################
#####    RETREIVING DATA AND FEATURES ENGENEERING   #######
###########################################################
dat_forex=forex_request('USD', 'EUR', frequency='d') # EUR for 1 USD
df=pd.merge(dat_forex,fred_merge(list(series_definition.keys())),on='Dates')
features_create(df)   # new data frame created with this line of code
############################################################


##########################################################
####    Function to automate machine learning        #####
##########################################################

def forex_learning(symbol2,symbol1='USD',endogeneous='Close',frequency='d',lag=2):
    import sklearn
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn import metrics
    
    dat_forex=forex_request(symbol1,symbol2, frequency=frequency) # EUR for 1 USD
    df=pd.merge(dat_forex,fred_merge(list(series_definition.keys())),on='Dates')
    features_create(df)   # new data frame created with this line of code
    
    # transform variables to lag values
    df2=lag_transform(df, n=lag)
    df2=pd.merge(df[endogeneous],df2.iloc[:,5:],on='Dates')
    
    # Defined label and features
    y=df2[endogeneous]
    X=df2[df2.columns[df2.columns!=endogeneous]]
    
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
   
    lr=linspace(0.05,1,num=10)
    rmse=[];mae=[]
    for r in lr:
        mod=GradientBoostingRegressor(learning_rate=r)
        mod.fit(X_train,y_train)
        y_pred=mod.predict(X_test)
        rmse.append(metrics.mean_squared_error(y_test, y_pred,squared=False)) # root mean square error (mean squqre error if squared=True)
        mae.append(metrics.mean_absolute_error(y_test,y_pred))
    return dict(zip(lr,rmse)),X_train,y_train,Xf
    
result=forex_learning(symbol2='EUR') 

#############
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
dd=df
dd=imput_missing.fit_transform(dd)
dd=pd.DataFrame(X);dd.columns=df.columns
fit1 =ExponentialSmoothing(
    dd.SP500+abs(min(dd.SP500))+0.5,
    seasonal_periods=7,
    trend="mul",
    seasonal="add",
    damped_trend=True,
    use_boxcox=True,
    initialization_method="estimated",
).fit()
fcast1 = fit1.forecast(15).rename(r"$\alpha=0.2$")
fcast1;fcast1.plot()
df.Open.append(fcast1)
op=df.Open
for i in range(0,3):
    fit1 = SimpleExpSmoothing(op, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
    fcast1 = fit1.forecast(1).rename(r"$\alpha=0.2$")
    op=op.append(fcast1)
pd.DataFrame(dd)

##########################
def forex_forecast(result,n_forcast=7): # res: result of forex_learning
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from datetime import datetime, date, timedelta

    mod=GradientBoostingRegressor(learning_rate=min(result[0], key=result[0].get))
    mod.fit(result[1],result[2])
    # Forecast values of X for n periods
    def fX_forecast(x): # x a data frame of X features to forecast in order to forecast y
        y=[]
        for i in x.columns:
            xf=ExponentialSmoothing(result[3].Day,#x[i],
            seasonal_periods=7,
            trend="mul",
            seasonal="add",
            damped_trend=True,
            use_boxcox=True,
            initialization_method="estimated"
            ).fit()
            y.append(fit1.forecast(15).rename(r"$\alpha=0.2$") )
        d=pd.DataFrame(y).transpose()
        # d.columns=result[3].columns
        last_date=datetime.strptime(result[3].index[result[3].shape[0]-1], '%Y-%m-%d')
        n_dates=[]
        for j in range(1,n_forcast+1):
            n_dates.append(last_date+timedelta(days=j))
        d.index=n_dates
        return d
    y_pred=mod.predict(result[3])
    return 

fX_forecast(result[3])
result[3]
