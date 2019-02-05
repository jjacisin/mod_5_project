# Code from notebooks will live here
import pandas as pd
import numpy as np

def format_data(df):
    ts_df = df.transpose()
    df_UE = ts_df.iloc[1:]
    df_UE.columns = ["seasonally_adjusted_unemployment_rate"]
    date_rng = pd.date_range(start='1/1/1948', end='12/31/2018', freq='MS')
    df_UE.index = date_rng
    return df_UE

def graph_creator(df):
    # pull x_values from index
    x_values = list(df.index.strftime("%Y/%m/%d"))
    # convert first  df column to y_values
    y_values = np.array(df.iloc[:,0:1]).flatten().tolist()
    return [{'x':x_values,'y':y_values,'name':'Initial Display'}]

df_ue_raw = pd.read_excel('dash_package/BLS_SA_Unemployment.xlsx',header=3)

#### PLOTABLE DATA

initial_display_ue = graph_creator(df_ue_raw)
