import pandas as pd
import numpy as np
from datetime import datetime

tickers = ['WIG', 'WIG20', 'PKN', 'KGH', 'plopln3m', 'TPE', 'ALE', 'TIM', 'MBK', 'PKO']


def data_from_period(start, end):
    dict_of_tickers = dict()
    for tick in tickers:
        data = pd.read_csv(f'{tick}.csv', index_col='Date', parse_dates=['Date'],
                date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data = data[start:end]
        data['zwykla_stopa_zwrotu'] = 100 * data['Close'].pct_change()
        data['logarytmiczna_stopa_zwrotu'] = 100 * (np.log(data.Close) - np.log(data.Close.shift(1)))
        data = data.dropna()

        if tick == 'plopln3m':
            data = data.drop(['Open', 'High', 'Low'], axis=1)
        else:
            data = data.drop(['Open', 'High', 'Low', 'Volume'], axis=1)
        dict_of_tickers[tick] = data
    return dict_of_tickers


def data_from_2007_to_2011():

    return data_from_period(start=datetime(2006, 12, 29), end=datetime(2012, 1, 1))


def data_from_2015_to_now():
    return data_from_period(start=datetime(2014, 12, 30), end=datetime(2021, 4, 1))


print(data_from_2007_to_2011())

#