# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:24:39 2022

@author: Diaraye
"""

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

df_fred=fred_merge(list(series_definition.keys()),'d')
# dat_forex=forex_request(symbol1,symbol2, frequency=frequency) # EUR for 1 USD
# df=pd.merge(dat_forex,fred_merge(list(series_definition.keys())),on='Dates')

df_forex=forex_request(symbol1='USD', symbol2='EUR', frequency='d') # EUR for 1 USD
# df=pd.merge(df_forex,fred_merge(list(series_definition.keys()),frequency=frequency),on='Dates')
df=pd.merge(df_forex,df_fred,on='Dates')
# return df
# result=forex_learning(symbol2='EUR')

############################33
features_create(df)   # new data frame created with this line of code
# transform variables to lag values
# df2=lag_transform(df, n=lag)
lag=2
endogeneous='Close'
df2=df.shift(lag).iloc[lag:,:]
df2=pd.merge(df[endogeneous],df2.iloc[:,5:],on='Dates')
# Defined label and features

# y=df[endogeneous].iloc[:-lag]
# y=df[endogeneous].iloc[lag:,:]
y=df[endogeneous][lag:]
# X=df2[df2.columns[df2.columns!=endogeneous]]
X=df2
# normalize the features
Xf=normalized_df(X)
# select featurest
features_selected=features_select(Xf)
X=features_selected[0]          # features selected
features_columns=features_selected[1]    # list of features names
###############
from sklearn.feature_selection import VarianceThreshold
selected = VarianceThreshold(threshold=0.02)
selected.fit_transform(Xf) # array of data of selected variables
#####
Xf=Xf[features_columns] # data frame of selected features to be used for forecasting new y
# replace missing value by mean value
from sklearn.impute import SimpleImputer
imput_missing = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imput_missing.fit_transform(X)
Xf=imput_missing.fit_transform(Xf) 
Xf=pd.DataFrame(Xf)
Xf.columns=features_columns;Xf.index=df2.index
# splitting data to train and test groups 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

X_train, X_test, y_train, y_test= train_test_split(X, y,train_size=0.25,random_state=0)  
lr=np.linspace(0.05,1,num=10)
rmse=[];mae=[]
for r in lr:
    mod=GradientBoostingRegressor(learning_rate=r)
    mod.fit(X_train,y_train)
    y_pred=mod.predict(X_test)
    rmse.append(metrics.mean_squared_error(y_test, y_pred,squared=False)) # root mean square error (mean squqre error if squared=True)
    mae.append(metrics.mean_absolute_error(y_test,y_pred))
rmse,mae
###########################################
last_date=datetime.strptime('2022-10-11','%Y-%m-%d')
da11=datetime.strftime(last_date+timedelta(days=1),'%Y-%m-%d')
da22=(datetime.strptime(da11,'%Y-%m-%d')).weekday();da22
def future_dates(x,n_forecast): # returns weekday
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
            fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)   # 1 is added to remove to replace 0 value by 1 for forecasting exchange rate using day value as features     
        last_date=datetime.strptime(fdate[len(fdate)-1], '%Y-%m-%d')  
    return fdate
future_dates(result, n_forecast=20)
datetime.strptime(result[3].index[0],'%Y-%m-%d').weekday()
datetime.strptime(result[3].index[1],'%Y-%m-%d').weekday()
datetime.strptime('2022-10-08','%Y-%m-%d').weekday() #samedi
datetime.strptime('2022-10-09','%Y-%m-%d').weekday() #dimanche
datetime.strptime(result[3].index[5],'%Y-%m-%d').weekday() #dimanche

datetime.strptime('2020-10-07','%Y-%m-%d')+timedelta(1)
#####
def future_dates0(x,n_forecast): # returns future dates in format '%Y-%m-%d'
    from datetime import datetime, date, timedelta
    last_date=datetime.strptime(x[3].index[len(x[3])-1], '%Y-%m-%d') # ok
    fdate=[];fdate1=[]
    for k in range(1,n_forecast+1): # k is the number of days to add to last date
        fdate.append(datetime.strftime(last_date+timedelta(days=k),'%Y-%m-%d')) # to have date in indicated format
        fdate1.append((datetime.strptime(fdate[k-1],'%Y-%m-%d')).weekday()+1)     
    return fdate

##############################333
def fred_features_create(x): # x is data frame of FRED series
    import itertools
    import pandas as pd
    pairs=[i for i in itertools.combinations(iterable=x.columns.tolist(), r=2)]
    ysum=[];yminus=[];yprod=[]
    for p in pairs:
        ysum.append(x[p[0]]+x[p[1]])
        yminus.append(abs(x[p[0]]-x[p[1]]))
        yprod.append(x[p[0]]*x[p[1]])
    df_sum=pd.DataFrame(ysum).transpose()
    df_minus=pd.DataFrame(yminus).transpose()
    df_prod=pd.DataFrame(yprod).transpose()
    df_sum.columns=[i[0]+'_sum_'+i[1] for i in pairs]
    df_minus.columns=[i[0]+'_minus_'+i[1] for i in pairs]
    df_prod.columns=[i[0]+'_prod_'+i[1] for i in pairs]
    
    df=pd.merge(pd.merge(pd.merge(x,df_sum,on='Dates'),df_minus,on='Dates'),df_prod,on='Dates')
    return df

fred_features_create(df.iloc[:,0:3]).head(2)
####################################################3
import seaborn as sns

dr=result[3].Close
ddf=pd.Series(index=['2022-10-17','2022-10-18'],data=[0.923,0.926])
dou=pd.concat([dr,ddf],axis=1)
sns.lineplot(data=dou,x=dou.index,y=dou.Close,ci=True)
sns.lineplot(data=dr,x=dr.Dates,y=dr.iloc[:,1])
plt.show()

##########33
# Define the confidence interval
ci = 0.1 * np.std(dou.Close) / np.mean(dou.Close)
fig=plt.figure()
plt.plot(dou.Close, color='red', lw=1)
plt.fill_between(dou.index, (dou.Close-ci), (dou.Close+ci), color='lightblue', alpha=0.5)
plt.show()
####################333
forX=forex_automated_forecast(symbol1='EUR',symbol2='USD',n_forecast=10)
dou.tail(20).plot()
