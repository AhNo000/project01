import datetime
import numpy as np
import pandas_datareader.data as web
import yfinance as yfin
from IPython.display import VimeoVideo

# 覆蓋yfinance的下載器以修正資料抓取
yfin.pdr_override()

# 設定開始和結束時間
start = datetime.date.today() - datetime.timedelta(days=5*365)
end = datetime.date.today()

# 從Yahoo Finance抓取資料
df = web.DataReader(["AMZN", "F", "BTC-USD"], start, end)['Adj Close']

# 檢視前幾行數據以確認數據的正確性
print(df.head())

