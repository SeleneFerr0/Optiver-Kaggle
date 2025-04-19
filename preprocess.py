#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sept 24 15:28:28 2021
First round trial - exploratory preprocessing
@author:selene
"""

import numpy as np
import os
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from utils import *
from pandas.api.types import CategoricalDtype
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#wdd = r'/Users/irena/Documents/optiver'
wdd =  r'D:\optiver'
os.chdir(wdd)
df_train = pd.read_csv(os.path.join(wdd, 'data', 'train.csv'))
cols = ['stock_id', 'time_id', 'target']
df_test = pd.read_csv(os.path.join(wdd, 'data', 'test.csv'))
cols = ['stock_id', 'time_id', 'target']
# 126 stocks
#cnt_var(raw,'stock_id')
tt = df_train[df_train.stock_id==0]

#%%
#rankings

targett(df_train,'target')
tgt_mean = df_train.groupby('stock_id')['target'].mean()
target_stds = df_train.groupby('stock_id')['target'].std()
tgt_mean_and_stds = pd.concat([tgt_mean, target_stds], axis=1)
tgt_mean_and_stds.columns = ['mean', 'std']
tgt_mean_and_stds.sort_values(by='mean', ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(32, 48))
ax.barh(y=np.arange(len(tgt_mean_and_stds)),
    width=tgt_mean_and_stds['mean'],xerr=tgt_mean_and_stds['std'], ecolor='black',align='center',capsize=3)

ax.set_yticks(np.arange(len(tgt_mean_and_stds)))
ax.set_yticklabels(tgt_mean_and_stds.index)
ax.set_xlabel('target', size=20, labelpad=15)
ax.set_ylabel('stock_id', size=20, labelpad=15)
ax.tick_params(axis='x', labelsize=20, pad=10)
ax.tick_params(axis='y', labelsize=20, pad=10)
ax.set_title('Mean r.vol', size=25, pad=20)
plt.show()

#%%
tgt_rmse = root_mean_squared_percentage_error(df_train['target'], np.repeat(df_train['target'].mean(), len(df_train)))
st_tgt_rmse = root_mean_squared_percentage_error(df_train['target'], df_train.groupby('stock_id')['target'].transform('mean'))
df_train['row_id'] = df_train['stock_id'].astype(str) + '_' + df_train['time_id'].astype(str)

fig, ax = plt.subplots(figsize=(32, 10))
ax.barh(y=np.arange(10),
    width=df_train.sort_values(by='target', ascending=True).tail(10)['target'],
    align='center', ecolor='black',)

ax.set_yticks(np.arange(10))
ax.set_yticklabels(df_train.sort_values(by='target', ascending=True).tail(10)['row_id'])
ax.set_xlabel('target', size=20, labelpad=15)
ax.set_ylabel('row_id', size=20, labelpad=15)
ax.tick_params(axis='x', labelsize=20, pad=10)
ax.tick_params(axis='y', labelsize=20, pad=10)
ax.set_title('Top 10 volatile', size=25, pad=20)
plt.show()
df_train.drop(columns=['row_id'], inplace=True)

df_train['row_id'] = df_train['stock_id'].astype(str) + '_' + df_train['time_id'].astype(str)
fig, ax = plt.subplots(figsize=(32, 10))
ax.barh(
    y=np.arange(10),
    width=df_train.sort_values(by='target', ascending=True).head(10)['target'],align='center',ecolor='black',)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(df_train.sort_values(by='target', ascending=True).head(10)['row_id'])
ax.set_xlabel('target', size=20, labelpad=15)
ax.set_ylabel('row_id', size=20, labelpad=15)
ax.tick_params(axis='x', labelsize=20, pad=10)
ax.tick_params(axis='y', labelsize=20, pad=10)
ax.set_title('Top 10 stable', size=25, pad=20)

plt.show()

df_train.drop(columns=['row_id'], inplace=True)

#%%
for stock_id in tqdm(sorted(df_train['stock_id'].unique())):
    df_raw_book = df_raw_data('train', stock_id)
    df_raw_book['wap1'] = (df_raw_book['bid_price1'] * df_raw_book['ask_size1'] + df_raw_book['ask_price1'] * df_raw_book['bid_size1']) /\
                      (df_raw_book['bid_size1'] + df_raw_book['ask_size1'])
    df_raw_book['wap2'] = (df_raw_book['bid_price2'] * df_raw_book['ask_size2'] + df_raw_book['ask_price2'] * df_raw_book['bid_size2']) /\
                      (df_raw_book['bid_size2'] + df_raw_book['ask_size2'])
    for agg in ['mean', 'std', 'min', 'max', realized_volatility]:
        bid_ask_price_r = df_raw_book.groupby('time_id')['bid_ask_price_ratio'].agg(agg)
        feature_name = agg.__name__ if callable(agg) else agg
        df_train.loc[df_train['stock_id'] == stock_id, f'book_bid_ask_price_ratio_{feature_name}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(bid_ask_price_ratio_aggregation)
        bid_ask_price_r = df_raw_book.groupby('time_id')['bid_ask_price_ratio'].agg(agg)
        feature_name = agg.__name__ if callable(agg) else agg
        df_train.loc[df_train['stock_id'] == stock_id, f'book_bid_ask_price_ratio_{feature_name}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(bid_ask_price_ratio_aggregation)
        wap_aggregation = df_raw_book.groupby('time_id')[f'wap{wap}'].agg(agg)
        feature_name = agg.__name__ if callable(agg) else agg
        df_train.loc[df_train['stock_id'] == stock_id, f'wap{wap}_{feature_name}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(wap_aggregation)         
    for wap in [1, 2]:
        df_raw_book[f'log_return_from_wap{wap}'] = np.log(df_raw_book[f'wap{wap}'] / df_raw_book.groupby('time_id')[f'wap{wap}'].shift(1))
        df_raw_book[f'squared_log_return_from_wap{wap}'] = df_raw_book[f'log_return_from_wap{wap}'] ** 2
        df_raw_book[f'realized_volatility_from_wap{wap}'] = np.sqrt(df_raw_book.groupby('time_id')[f'squared_log_return_from_wap{wap}'].transform('sum'))
        df_raw_book.drop(columns=[f'squared_log_return_from_wap{wap}'], inplace=True)            
        realized_volatilities = df_raw_book.groupby('time_id')[f'realized_volatility_from_wap{wap}'].first().to_dict()
        df_train.loc[df_train['stock_id'] == stock_id, f'realized_volatility_from_wap{wap}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(realized_volatilities)
        
df_train['stock_id'] = df_train['stock_id'].astype(str)
#df_train.to_csv(os.path.join(wdd, 'df_aLL_st.csv'))
#%%
### stock id = 18 / 43
df_raw_book = df_raw_dataload('train', '43')
df_raw_book['wap1'] = (df_raw_book['bid_price1'] * df_raw_book['ask_size1'] + df_raw_book['ask_price1'] * df_raw_book['bid_size1']) /(df_raw_book['bid_size1'] + df_raw_book['ask_size1'])
df_raw_book['wap2'] = (df_raw_book['bid_price2'] * df_raw_book['ask_size2'] + df_raw_book['ask_price2'] * df_raw_book['bid_size2']) /(df_raw_book['bid_size2'] + df_raw_book['ask_size2'])
for wap in [1, 2]:
    df_raw_book[f'log_return_from_wap{wap}'] = np.log(df_raw_book[f'wap{wap}'] / df_raw_book.groupby('time_id')[f'wap{wap}'].shift(1))
    df_raw_book[f'squared_log_return_from_wap{wap}'] = df_raw_book[f'log_return_from_wap{wap}'] ** 2
    df_raw_book[f'realized_volatility_from_wap{wap}'] = np.sqrt(df_raw_book.groupby('time_id')[f'squared_log_return_from_wap{wap}'].transform('sum'))
    df_raw_book.drop(columns=[f'squared_log_return_from_wap{wap}'], inplace=True)            
    realized_volatilities = df_raw_book.groupby('time_id')[f'realized_volatility_from_wap{wap}'].first().to_dict()
    df_train.loc[df_train['stock_id'] == '43', f'realized_volatility_from_wap{wap}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(realized_volatilities)

vol_order(df_train,stock_id='43', time_id=30128)
for feat in df_train.columns[6:]:
    feat_eda(feat)

