# for stock price calculations

# Essentially you divide the change in price by a volatility measure. This helps you see if the price has moved a lot in the last 200 days relative to how volatile the stock is. Assuming you have 2 stocks that have both increased by $100 over the last 200 days. Calculate the volatility of both using any measure (in this case we use ATR (see wikipedia link) which you can think about as the size of an average candle or daily bar for the stock. Assuming you have a stock that moves on average $1 per day and a stock that moves on average $10 per day.
# Before dividing by ATR, both stocks show a $100 movement over the last 200 days. But once you divide by ATR, you see stock 1 has moved 100 times its daily movement, while stock 2 moved only 10 times its daily movement over the same time period. Hence it seems like stock 1 has had a more significant move relevant to stock 2, which you only see by dividing by the ATR.
# Can divide percent change by ATR (Average True Range: https://en.wikipedia.org/wiki/Average_true_range) to account for differences between high and low volatility stocks


# import datetime
# from pandas.io.data import DataReader
from datetime import datetime, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
import pandas_datareader as pdr
# import fix_yahoo_finance as yf
from dateutil.relativedelta import relativedelta

# today = datetime.datetime.now().strftime('%Y-%m-%d')
# print(today)

def add_years(d, years):
    try:
        return d.replace(year = d.year + years)
    except ValueError:
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))

def add_months(d, num_months):
    return d + relativedelta(months=+num_months)


date2 = datetime(2018,6,14)
date3 = datetime(2018,2,1)
# print(add_years(date2, 1))
print(add_months(date2, 9))
print(date2)
print(date3)

# google = pdr.get_data_yahoo("GOOGL",
#                             start = "2018-02-01",
#                             end = "2018-02-01")

google = pdr.get_data_yahoo("GOOGL", start = datetime(2018,2,1), end = datetime(2018,2,1))
print(google.columns)
print(google.iloc[0]['Adj Close'])
