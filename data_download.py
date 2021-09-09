from urllib.request import urlretrieve
from datetime import datetime
import pandas as pd

tickers = ['WIG', 'WIG20', 'PKN', 'KGH', 'MIL', 'TPE', 'ING', 'PGN', 'MBK', 'PKO', 'BOS']
interval = 'd'

for ticker in tickers:
    print(f'teraz {ticker}')
    url = f'https://stooq.com/q/d/l/?s={ticker}&i={interval}'
    csv_file = ticker + '.csv'
    urlretrieve(url, csv_file)
    data = pd.read_csv(csv_file, index_col='Date', parse_dates=['Date'],
                date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    print(f'ściągnięto {ticker}')
