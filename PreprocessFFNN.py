import os
import gc
import glob
from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy.matlib
target_name = 'target'
err_folds = {}
import random
random.seed(99)
import tensorflow as tf
tf.random.set_seed(99)
from tensorflow import keras
import numpy as np
from keras import backend as K
from sklearn.cluster import KMeans
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


data_dir = '/Users/carol/OneDrive/Desktop/Fall 2021/Data Mining/Final Project/'
def calc_wap1(ffnndf):
    wap = (ffnndf['bid_price1'] * ffnndf['ask_size1'] + ffnndf['ask_price1'] * ffnndf['bid_size1']) / (ffnndf['bid_size1'] + ffnndf['ask_size1'])
    return wap
def calc_wap2(ffnndf):
    wap = (ffnndf['bid_price2'] * ffnndf['ask_size2'] + ffnndf['ask_price2'] * ffnndf['bid_size2']) / (ffnndf['bid_size2'] + ffnndf['ask_size2'])
    return wap
def log_return(prices):
    return np.log(prices).diff()
def realized_volatility(ret):
    return np.sqrt(np.sum(ret**2))
def count_unique(ret):
    return len(np.unique(ret))
def read_train_test():
    train = pd.read_csv('/Users/carol/OneDrive/Desktop/Fall 2021/Data Mining/Final Project/newtrain.csv')
    test = pd.read_csv('/Users/carol/OneDrive/Desktop/Fall 2021/Data Mining/Final Project/newtest.csv')
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    return train, test

def order_preprocess(file_path):
    ffnndf = pd.read_parquet(file_path)
    ffnndf['wap1'] = calc_wap1(ffnndf)
    ffnndf['wap2'] = calc_wap2(ffnndf)
    ffnndf['log_return1'] = ffnndf.groupby(['time_id'])['wap1'].apply(log_return)
    ffnndf['log_return2'] = ffnndf.groupby(['time_id'])['wap2'].apply(log_return)
    ffnndf['wap_balance'] = abs(ffnndf['wap1'] - ffnndf['wap2'])
    ffnndf['price_spread'] = (ffnndf['ask_price1'] - ffnndf['bid_price1']) / ((ffnndf['ask_price1'] + ffnndf['bid_price1']) / 2)
    ffnndf['price_spread2'] = (ffnndf['ask_price2'] - ffnndf['bid_price2']) / ((ffnndf['ask_price2'] + ffnndf['bid_price2']) / 2)
    ffnndf['bid_spread'] = ffnndf['bid_price1'] - ffnndf['bid_price2']
    ffnndf['ask_spread'] = ffnndf['ask_price1'] - ffnndf['ask_price2']
    ffnndf["bid_ask_spread"] = abs(ffnndf['bid_spread'] - ffnndf['ask_spread'])
    ffnndf['sum_volume'] = (ffnndf['ask_size1'] + ffnndf['ask_size2']) + (ffnndf['bid_size1'] + ffnndf['bid_size2'])
    ffnndf['volume_imbal'] = abs((ffnndf['ask_size1'] + ffnndf['ask_size2']) - (ffnndf['bid_size1'] + ffnndf['bid_size2']))
    
    create_feature_dict = {
        'wap1': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'wap_balance': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'price_spread2':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'sum_volume':[np.sum, np.mean, np.std],
        'volume_imbal':[np.sum, np.mean, np.std],
        "bid_ask_spread":[np.sum, np.mean, np.std],
    }
    
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        df_feature = ffnndf[ffnndf['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis = 1, inplace = True)
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    return df_feature

def trade_preprocess(file_path):
    ffnndf = pd.read_parquet(file_path)
    ffnndf['log_return'] = ffnndf.groupby('time_id')['price'].apply(log_return)
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum, realized_volatility, np.mean, np.std, np.max, np.min],
        'order_count':[np.mean,np.sum,np.max],
    }
    
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        df_feature = ffnndf[ffnndf['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_450 = get_stats_window(seconds_in_bucket = 450, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_150 = get_stats_window(seconds_in_bucket = 150, add_suffix = True)
    
    def trend(price, vol):    
        df_diff = np.diff(price)
        val = (df_diff/price[1:])*100
        power = np.sum(val*vol[1:])
        return(power)
    
    lst = []
    for n_time_id in ffnndf['time_id'].unique():
        df_id = ffnndf[ffnndf['time_id'] == n_time_id]        
        trendV = trend(df_id['price'].values, df_id['size'].values) 
        df_max =  np.sum(np.diff(df_id['price'].values) > 0)
        df_min =  np.sum(np.diff(df_id['price'].values) < 0)     
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        abs_diff = np.median(np.abs( df_id['price'].values - np.mean(df_id['price'].values)))        
        energy = np.mean(df_id['price'].values**2)
        iqr_p = np.percentile(df_id['price'].values,75) - np.percentile(df_id['price'].values,25)
        abs_diff = np.median(np.abs( df_id['size'].values - np.mean(df_id['size'].values)))        
        energy_sum_squared = np.sum(df_id['size'].values**2)
        abs_perc_diff = np.percentile(df_id['size'].values,75) - np.percentile(df_id['size'].values,25)
        
        lst.append({'time_id':n_time_id,'trend':trendV,
                    'f_max':f_max,'f_min':f_min,
                    'df_max':df_max,'df_min':df_min,
                   'abs_diff':abs_diff,'energy':energy,
                    'iqr_p':iqr_p,'abs_diff':abs_diff,
                    'energy_sum_squared':energy_sum_squared,'abs_perc_diff':abs_perc_diff})
    
    df_lr = pd.DataFrame(lst)
    df_feature = df_feature.merge(df_lr, how = 'left', left_on = 'time_id_', right_on = 'time_id')
    df_feature = df_feature.merge(df_feature_450, how = 'left', left_on = 'time_id_', right_on = 'time_id__450')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_150, how = 'left', left_on = 'time_id_', right_on = 'time_id__150')
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150','time_id'], axis = 1, inplace = True)
    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    return df_feature

def stock_time_window(ffnndf):
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 
                'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 
                'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 
                'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']
    df_stock_id = ffnndf.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')
    df_time_id = ffnndf.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    ffnndf = ffnndf.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    ffnndf = ffnndf.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    ffnndf.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    return ffnndf

def preprocess(stock_ids, is_train = True):
    def for_joblib(stock_id):
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        df_tmp = pd.merge(order_preprocess(file_path_book), trade_preprocess(file_path_trade), on = 'row_id', how = 'left')
        return df_tmp
    ffnndf = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in stock_ids)
    ffnndf = pd.concat(ffnndf, ignore_index = True)
    return ffnndf
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


train, test = read_train_test()
train_stock_ids = train['stock_id'].unique()
train_ = preprocess(train_stock_ids, is_train = True)
train = train.merge(train_, on = ['row_id'], how = 'left')

test_stock_ids = test['stock_id'].unique()
test_ = preprocess(test_stock_ids, is_train = False)
test = test.merge(test_, on = ['row_id'], how = 'left')
train = stock_time_window(train)
test = stock_time_window(test)

train['size_sum'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique'] )
test['size_sum'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique'] )
train['size_sum_450'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_450'] )
test['size_sum_450'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_450'] )
train['size_sum_300'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_300'] )
test['size_sum_300'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_300'] )
train['size_sum_150'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_150'] )
test['size_sum_150'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_150'] )
train['size_sum2'] = np.sqrt( 1/ train['trade_order_count_sum'] )
test['size_sum2'] = np.sqrt( 1/ test['trade_order_count_sum'] )
train['size_sum2_450'] = np.sqrt( 0.25/ train['trade_order_count_sum'] )
test['size_sum2_450'] = np.sqrt( 0.25/ test['trade_order_count_sum'] )
train['size_sum2_300'] = np.sqrt( 0.5/ train['trade_order_count_sum'] )
test['size_sum2_300'] = np.sqrt( 0.5/ test['trade_order_count_sum'] )
train['size_sum2_150'] = np.sqrt( 0.75/ train['trade_order_count_sum'] )
test['size_sum2_150'] = np.sqrt( 0.75/ test['trade_order_count_sum'] )
train['size_sum2_d'] = train['size_sum2_450'] - train['size_sum2']
test['size_sum2_d'] = test['size_sum2_450'] - test['size_sum2']
train_p = pd.read_csv('/Users/carol/OneDrive/Desktop/Fall 2021/Data Mining/Final Project/newtrain.csv')
train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')