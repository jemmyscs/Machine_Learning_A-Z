from numpy import *
import numpy as np
import tushare as ts
import pandas as pd
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import yfinance as yf

# Get the stock data from tushare
def get_stock_data(code,pred_days):
    # df_raw = ts.get_k_data(code)
    # df_raw = yf.download(code, start='2022-01-01', end='2024-07-26')
    df_raw = yf.download(code, start=None, end='2024-07-26')
    # Classification
    label = ['']*len(df_raw['Close'])
    for i in range(len(df_raw['Close'])-pred_days):
        if (df_raw['Close'][i + pred_days] - df_raw['Close'][i]) > 0:
            label[i] = 1
        else:
            label[i] = -1
    # Save to csv file
    df_raw['LABEL'] = label
    df_raw.to_csv('raw_stock_data.csv')
    return 'raw_stock_data.csv'

def exponential_smoothing(alpha, s):
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(1, len(s2)):
        s2[i] = alpha*float(s[i])+(1-alpha)*float(s2[i-1])
    return s2

# preprocess the stock data with exponential_smoothing
def em_stock_data(pathfile,alpha):
    df = pd.read_csv(pathfile)
    es_open = pd.DataFrame(exponential_smoothing(alpha,np.array(df['Open'])))
    es_close = pd.DataFrame(exponential_smoothing(alpha, np.array(df['Close'])))
    es_high = pd.DataFrame(exponential_smoothing(alpha, np.array(df['High'])))
    es_low = pd.DataFrame(exponential_smoothing(alpha, np.array(df['Low'])))
    df['Open'],df['Close'],df['High'],df['Low'] = es_open,es_close,es_high,es_low
    df.to_csv('em_stock_data.csv')
    return str('em_stock_data.csv')

# preprocess the stock data with calc_technical_indicators
def calc_technical_indicators(filepath):
    df = pd.read_csv(filepath, index_col='Date')
    # Simple Moving Average SMA 简单移动平均$
    df['SMA5'] = talib.MA(df['Close'], timeperiod=5)
    df['SMA10'] = talib.MA(df['Close'], timeperiod=10)
    df['SMA20'] = talib.MA(df['Close'], timeperiod=20)
    # Williams Overbought/Oversold Index WR 威廉指标
    df['WR14'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['WR18'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=18)
    df['WR22'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=22)
    # Moving Average Convergence / Divergence MACD 指数平滑移动平均线
    DIFF1, DEA1, df['MACD9'] = talib.MACD(np.array(df['Close']), fastperiod=12, slowperiod=26, signalperiod=9)
    DIFF2, DEA2, df['MACD10'] = talib.MACD(np.array(df['Close']), fastperiod=14, slowperiod=28, signalperiod=10)
    DIFF3, DEA3, df['MACD11'] = talib.MACD(np.array(df['Close']), fastperiod=16, slowperiod=30, signalperiod=11)
    df['MACD9'] = df['MACD9'] * 2
    df['MACD10'] = df['MACD10'] * 2
    df['MACD11'] = df['MACD11'] * 2
    # Relative Strength Index RSI 相对强弱指数
    df['RSI15'] = talib.RSI(np.array(df['Close']), timeperiod=15)
    df['RSI20'] = talib.RSI(np.array(df['Close']), timeperiod=20)
    df['RSI25'] = talib.RSI(np.array(df['Close']), timeperiod=25)
    df['RSI30'] = talib.RSI(np.array(df['Close']), timeperiod=30)
    # Stochastic Oscillator Slow STOCH 常用的KDJ指标中的KD指标
    df['STOCH'] = \
    talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3,
                slowd_matype=0)[1]
    # On Balance Volume OBV 能量潮
    df['OBV'] = talib.OBV(np.array(df['Close']), df['Volume'])
    # Simple moving average SMA 简单移动平均
    df['SMA15'] = talib.SMA(df['Close'], timeperiod=15)
    df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA25'] = talib.SMA(df['Close'], timeperiod=25)
    df['SMA30'] = talib.SMA(df['Close'], timeperiod=30)
    # Money Flow Index MFI MFI指标
    df['MFI14'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['MFI18'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=18)
    df['MFI22'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=22)
    # Ultimate Oscillator UO 终极指标
    df['UO7'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['UO8'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=8, timeperiod2=16, timeperiod3=22)
    df['UO9'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=9, timeperiod2=18, timeperiod3=26)
    # Rate of change Percentage ROCP 价格变化率
    df['ROCP'] = talib.ROCP(df['Close'], timeperiod=10)
    df.to_csv('final_stock_data.csv')
    return 'final_stock_data.csv'

# preprocess the stock data with normalization and split data
def normalization(filepath,features):
    df= pd.read_csv(filepath)
    df = df[33:(len(df['Volume']) - pred_days)]
    # normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    for i in range(len(features)):
        df[features[i]] = min_max_scaler.fit_transform(np.reshape(df[features[i]] ,(-1,1)))
    # split data
    df_len = len(df)
    df_train = df[:int(df_len * 0.8)]
    df_valid = df[int(df_len * 0.8):int(df_len * 0.9)]
    df_test = df[int(df_len * 0.9):]
    df_train.to_csv('train.csv')
    df_valid.to_csv('valid.csv')
    df_test.to_csv('test.csv')
    return 'train.csv','valid.csv','test.csv'


def random_forest_model(train_filepath,valid_filepath,test_filepath,features):
    df_train = pd.read_csv(train_filepath)
    df_valid = pd.read_csv(valid_filepath)
    df_test = pd.read_csv(test_filepath)
    # set hyper-parameter
    sample_leaf_options = 7
    n_estimators_options = 7
    alg = RandomForestClassifier(criterion='gini',bootstrap=True,min_samples_leaf=sample_leaf_options, n_estimators=n_estimators_options, random_state=50)
    alg.fit(df_train[features],df_train['LABEL'])
    predict = alg.predict(df_test[features])
    features_degree = sorted(zip(map(lambda x: round(x, 4), alg.feature_importances_),df_train[features]), reverse=True)
    pred_accuracy = (df_test['LABEL'] == predict).mean()

    X = df_train[features]
    y = df_train[["Close"]]
    regr = RandomForestRegressor(max_depth=21, random_state=0)
    regr.fit(X, y)
    X_test = df_test[features]
    print(regr.predict(X_test))

    return pred_accuracy,features_degree

if __name__=='__main__':
    code = 'AAPL'
    alpha = 0.7
    pred_days = 15
    features = ["Open", "Close", "High", "Low", "Volume", "SMA5", "WR14", "MACD9", "RSI15", "MFI14", "UO7", "ROCP"]
    raw_filepath = get_stock_data(code=code,pred_days=pred_days)
    em_filepath = em_stock_data(pathfile=raw_filepath, alpha=alpha)
    final_filepath = calc_technical_indicators(filepath=em_filepath)
    train_filepath,valid_filepath,test_filepath = normalization(filepath=final_filepath,features=features)
    pred_accuracy, features_degree = random_forest_model(train_filepath,valid_filepath,test_filepath,features)
    print('pred_accuracy: ',pred_accuracy)
    print('features_degree: ',features_degree)