import warnings
warnings.filterwarnings('ignore')  # Hide warnings
import datetime as dt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
#from mplfinance import candlestick_ohlc
import matplotlib.dates as mdates
import streamlit as st

import plotly
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly


#st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Stock forecast dashboard')
st.write ("Developed by Akash Shankar - Intern Virtusa")


com = st.text_input("Enter the Stock Code of company","AAPL")

'You Enterted the company code: ', com

st_date= st.text_input("Enter Starting date as YYYY-MM-DD", "2000-01-10")

'You Enterted the starting date: ', st_date

end_date= st.text_input("Enter Ending date as YYYY-MM-DD", "2000-01-20")

'You Enterted the ending date: ', end_date

n_years = st.slider("Years Of Prediction:", 0, 3)
period = n_years * 365

df = web.DataReader(com, 'yahoo', st_date, end_date)
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)

df

'The Stock Close Values over Time: '
st.line_chart(df["Close"])

'The Stock Open Values over Time: '
st.line_chart(df["Open"])

#moving averages
st.title('Moving Average')

'Stock Data based on Moving Average'

mov_avg = st.text_input("Enter number of days Moving Average: ", "20")

"Value Entered is :",mov_avg

df["mavg_close"] = df['Close'].rolling(window=int(mov_avg),min_periods=0).mean()

'Plot of Stock Closing Value for '+ mov_avg + "days of moving average"

st.line_chart(df[["mavg_close","Close"]])


df.reset_index(inplace=True)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Stock_Open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock_Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = df[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())


st.write('Forecasted Data: ')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('The Forecast Components: ')
fig2 = m.plot_components(forecast)
st.write(fig2)