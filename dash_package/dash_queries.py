# Code from notebooks will live here
from dash_package import app
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def format_data(df):
    ts_df = df.transpose()
    df_UE = ts_df.iloc[1:]
    df_UE.columns = ["seasonally_adjusted_unemployment_rate"]
    date_rng = pd.date_range(start='1/1/1948', end='12/31/2018', freq='MS')
    df_UE.index = date_rng
    return df_UE

def graph_creator(df,name='Initial Display'):
    # pull x_values from index
    x_values = list(df.index.strftime("%Y/%m/%d"))
    # convert first  df column to y_values
    y_values = np.array(df.iloc[:,0:1]).flatten().tolist()
    return {'x':x_values,'y':y_values,'name':name}


def use_rol_mean(pd_series, window=12):
    rolmean = pd_series.rolling(window).mean()
    data_minus_rolmean = pd_series - rolmean
    # Adding in rolmean creates NaNs for first year
    data_minus_rolmean.dropna(inplace=True)
    return data_minus_rolmean

def test_stationarity(timeseries,window=12):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean - {}'.format(str(window)))
    std = plt.plot(rolstd, color='black', label = 'Rolling Std - {}'.format(str(window)))
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Pull in Data
df_ue_raw = pd.read_excel('dash_package/BLS_SA_Unemployment.xlsx',header=3)
df_ue_orig = format_data(df_ue_raw)
df_ue_in_scope = df_ue_orig['2005-01-01':]
df_ue_std = use_rol_mean(df_ue_in_scope)

#### PLOTABLE DATA

ue_initial_display = graph_creator(df_ue_orig)
ue_in_scope_display = graph_creator(df_ue_in_scope,"Original")
ue_standardized_data = graph_creator(df_ue_std,"12M Adjusted")
