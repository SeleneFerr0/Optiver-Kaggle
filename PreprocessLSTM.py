import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import copy
from sklearn.model_selection import KFold
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


data_dir = '/Users/carol/OneDrive/Desktop/Fall 2021/Data Mining/Final Project/'

def calc_wap(LSTMdf):
    wap = (LSTMdf['bid_price1'] * LSTMdf['ask_size1'] + LSTMdf['ask_price1'] * LSTMdf['bid_size1'])/(LSTMdf['bid_size1'] + LSTMdf['ask_size1'])
    return wap
def calc_wap2(LSTMdf):
    wap = (LSTMdf['bid_price2'] * LSTMdf['ask_size2'] + LSTMdf['ask_price2'] * LSTMdf['bid_size2'])/(LSTMdf['bid_size2'] + LSTMdf['ask_size2'])
    return wap
def calc_wap3(LSTMdf):
    wap = (LSTMdf['bid_price2'] * LSTMdf['bid_size2'] + LSTMdf['ask_price2'] * LSTMdf['ask_size2']) / (LSTMdf['bid_size2']+ LSTMdf['ask_size2'])
    return wap
def log_return(prices):
    return np.log(prices).diff() 
def realized_volatility(ret):
    return np.sqrt(np.sum(ret**2))
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
def count_unique(ret):
    return len(np.unique(ret))
book_train = pd.read_parquet(data_dir + "book_train.parquet/stock_id=0")

def order_preprocess(file_path):
    LSTMdf = pd.read_parquet(file_path)
    LSTMdf['wap'] = calc_wap(LSTMdf)
    LSTMdf['wap2'] = calc_wap2(LSTMdf)
    LSTMdf['wap3'] = calc_wap3(LSTMdf)

    LSTMdf['log_return'] = LSTMdf.groupby('time_id')['wap'].apply(log_return)
    LSTMdf['log_return2'] = LSTMdf.groupby('time_id')['wap2'].apply(log_return)
    LSTMdf['log_return3'] = LSTMdf.groupby('time_id')['wap3'].apply(log_return)

    LSTMdf['wap_balance'] = abs(LSTMdf['wap'] - LSTMdf['wap2'])

    LSTMdf['price_spread'] = (LSTMdf['ask_price1'] - LSTMdf['bid_price1']) / ((LSTMdf['ask_price1'] + LSTMdf['bid_price1'])/2)
    LSTMdf['bid_spread'] = LSTMdf['bid_price1'] - LSTMdf['bid_price2']
    LSTMdf['ask_spread'] = LSTMdf['ask_price1'] - LSTMdf['ask_price2']

    LSTMdf['sum_volume'] = (LSTMdf['ask_size1'] + LSTMdf['ask_size2']) + (LSTMdf['bid_size1'] + LSTMdf['bid_size2'])

    LSTMdf['volume_imbal'] = abs((LSTMdf['ask_size1'] + LSTMdf['ask_size2']) - (LSTMdf['bid_size1'] + LSTMdf['bid_size2']))

    create_feature_dict = {
        'log_return':[realized_volatility],
        'log_return2':[realized_volatility],
        'log_return3':[realized_volatility],
        'wap_balance':[np.mean],
        'price_spread':[np.mean],
        'bid_spread':[np.mean],
        'ask_spread':[np.mean],
        'volume_imbal':[np.mean],
        'sum_volume':[np.mean],
        'wap':[np.mean],
            }

    df_feature = pd.DataFrame(LSTMdf.groupby(['time_id']).agg(create_feature_dict)).reset_index()
    
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]

    time_windows = [300]
    for sec in time_windows:
        sec = 600 - sec 
        df_feature_sec = pd.DataFrame(LSTMdf.query(f'seconds_in_bucket >= {sec}').groupby(['time_id']).agg(create_feature_dict)).reset_index()
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns] 
        df_feature_sec = df_feature_sec.add_suffix('_' + str(sec))
        df_feature = pd.merge(df_feature,df_feature_sec,how='left',left_on='time_id_',right_on=f'time_id__{sec}')
        df_feature = df_feature.drop([f'time_id__{sec}'],axis=1)

    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['time_id_'],axis=1)
    return df_feature

%%time
file_path = data_dir + "book_train.parquet/stock_id=0"
order_preprocess(file_path)
trade_train = pd.read_parquet(data_dir + "trade_train.parquet/stock_id=0")

def trade_preprocess(file_path):
    LSTMdf = pd.read_parquet(file_path)
    LSTMdf['log_return'] = LSTMdf.groupby('time_id')['price'].apply(log_return)
    
    
    aggregate_dictionary = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.mean],
    }
    
    df_feature = LSTMdf.groupby('time_id').agg(aggregate_dictionary)
    
    df_feature = df_feature.reset_index()
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    time_windows = [300]
    
    for sec in time_windows:
        sec = 600 - sec
    
        df_feature_sec = LSTMdf.query(f'seconds_in_bucket >= {sec}').groupby('time_id').agg(aggregate_dictionary)
        df_feature_sec = df_feature_sec.reset_index()
        
        df_feature_sec.columns = ['_'.join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix('_' + str(sec))
        
        df_feature = pd.merge(df_feature,df_feature_sec,how='left',left_on='time_id_',right_on=f'time_id__{sec}')
        df_feature = df_feature.drop([f'time_id__{sec}'],axis=1)
    
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature = df_feature.drop(['trade_time_id_'],axis=1)
    
    return df_feature
%%time
file_path = data_dir + "trade_train.parquet/stock_id=0"
trade_preprocess(file_path)


def preprocess(stock_ids, is_train = True):
    from joblib import Parallel, delayed
    LSTMdf = pd.DataFrame()
    
    def for_joblib(stock_id):
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
            
        df_tmp = pd.merge(order_preprocess(file_path_book),trade_preprocess(file_path_trade),on='row_id',how='left')
     
        return pd.concat([LSTMdf,df_tmp])
    
    LSTMdf = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in stock_ids
        )

    LSTMdf =  pd.concat(LSTMdf,ignore_index = True)
    return LSTMdf

stock_ids = [0,1]
preprocess(stock_ids, is_train = True)

%%time
df_train = preprocess(stock_ids= train_ids, is_train = True)
train_ids = train.stock_id.unique()

train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
train = train[['row_id','target']]
df_train = train.merge(df_train, on = ['row_id'], how = 'left')
df_train.fillna(0, inplace=True)


test1 = pd.read_csv(data_dir + 'LSTMtest.csv')
test = copy.copy(test1)
test = test[['stock_id', 'time_id']].astype(str) 
test["row_id"] = test[['stock_id', 'time_id']].agg('-'.join, axis=1)
test_ids = test.stock_id.unique()

%%time
df_test = preprocess(stock_ids= test_ids, is_train = True)

df_test = test.merge(df_test, on = ['row_id'], how = 'left')
df_train['stock_id'] = df_train['row_id'].apply(lambda x:x.split('-')[0])
df_test['stock_id'] = df_test['row_id'].apply(lambda x:x.split('-')[0])

stock_id_target_mean = df_train.groupby('stock_id')['target'].mean() 
df_test['stock_id_target_mapped'] = df_test['stock_id'].map(stock_id_target_mean) # test_set
temp = np.repeat(np.nan, df_train.shape[0])
kf = KFold(n_splits = 20, shuffle=True,random_state = 99)
for idx_1, idx_2 in kf.split(df_train):
    target_mean = df_train.iloc[idx_1].groupby('stock_id')['target'].mean()
    temp[idx_2] = df_train['stock_id'].iloc[idx_2].map(target_mean)
df_train['stock_id_target_mapped'] = temp

df_train['stock_id'] = df_train['stock_id'].astype(int)
df_test['stock_id'] = df_test['stock_id'].astype(int)
X= df_train.drop(['row_id','target'],axis=1)
y= df_train['target']
kf = KFold(n_splits=25, random_state=99, shuffle=True)
oof = pd.DataFrame()                 
models = []                          
scores = 0 