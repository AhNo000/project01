import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yfin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 覆蓋yfinance的下載器以修正資料抓取
yfin.pdr_override()

# 設定開始和結束時間
start = datetime.date.today() - datetime.timedelta(days=5)
end = datetime.date.today()

# 從Yahoo Finance抓取資料
data = web.DataReader(["TSM"], start, end)['Adj Close']

# 檢視前幾行數據以確認數據的正確性
print(data.head())


# 資料準備和預處理

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Adj Close']])

# 創建數據窗口
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# 分割數據集
train_size = int(len(data_scaled) * 0.7)
test_size = len(data_scaled) - train_size
train, test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]
look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# LSTM模型
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# XGBoost模型
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
model_xgb.fit(X_train[:,0], y_train)

# 預測
predictions_lstm = model.predict(np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1])))
predictions_xgb = model_xgb.predict(X_test[:,0])

# 結果融合
predictions_final = 0.5 * predictions_lstm.flatten() + 0.5 * predictions_xgb

# 評估
mse = mean_squared_error(y_test, predictions_final)
print('MSE: ', mse)
