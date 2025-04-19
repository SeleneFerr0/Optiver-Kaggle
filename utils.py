# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 05:18:53 2021
utility funcions, plots EDA
@author: selene
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from scipy.stats import probplot


def cnt_var(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + str(len(df[df[var]== v])))


def cnt_per(df, var):
    var_lst = list(df[var].unique())
    print(var)
    for v in var_lst:
        print(str(v) + ' ' + "{:.0%}".format((len(df[df[var]== v])/len(df))))


def plot_corr(df,size=20):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    
def log_return(x):
    return np.log(x).diff()

def realized_volatility(x):
    return np.sqrt(np.sum(log_return(x) ** 2))

def root_mean_squared_percentage_error(y_true, y_pred, epsilon=1e-10):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + epsilon))))
    return rmspe

def targett(df_dat, target):
    fig, axes = plt.subplots(ncols=2, figsize=(24, 8), dpi=100)
    sns.kdeplot(df_dat[target], label=target, ax=axes[0])
    axes[0].axvline(df_dat[target].mean(), label=f'{target} Mean', color='r', linewidth=2, linestyle='--')
    axes[0].axvline(df_dat[target].median(), label=f'{target} Median', color='b', linewidth=2, linestyle='--')
    probplot(df_dat[target], plot=axes[1])
    axes[0].legend(prop={'size': 16})
    
    for i in range(2):
        axes[i].tick_params(axis='x', labelsize=12.5, pad=10)
        axes[i].tick_params(axis='y', labelsize=12.5, pad=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    axes[0].set_title(f'{target} Distribution', fontsize=20, pad=15)
    axes[1].set_title(f'{target} Probability', fontsize=20, pad=15)

    plt.show()
    
def df_raw_dataload(dataset, stock_id,forward_fill=False):
    df_varty = {'time_id': np.uint16,'time_group': np.uint16,'bid_price1': np.float32,
        'ask_price1': np.float32,'bid_price2': np.float32,'ask_price2': np.float32,
        'bid_size1': np.uint32,'ask_size1': np.uint32,'bid_size2': np.uint32,'ask_size2': np.uint32,}
    df_dat = pd.read_parquet(f'D:/optiver/data/book_{dataset}.parquet/stock_id={stock_id}')
    for column, dttype in df_varty.items():
        df_dat[column] = df_dat[column].astype(dttype)
    df_dat.sort_values(by=['time_id', 'time_group'], inplace=True)
    if forward_fill:
        df_dat = df_dat.set_index(['time_id', 'time_group'])
        df_dat = df_dat.reindex(pd.MultiIndex.from_product([df_dat.index.levels[0], np.arange(0, 600)], names=['time_id', 'time_group']), method='ffill')
        df_dat.reset_index(inplace=True)
    return df_dat

def vol_order(df_dat,stock_id, time_id):
    df_dat = df_raw_dataload('train', stock_id, sort=True, forward_fill=True)
    df_dat = df_dat.set_index('time_group')
    df_dat['wap1'] = (df_dat['bid_price1'] * df_dat['ask_size1'] + df_dat['ask_price1'] * df_dat['bid_size1']) /\
                      (df_dat['bid_size1'] + df_dat['ask_size1'])
    df_dat['wap2'] = (df_dat['bid_price2'] * df_dat['ask_size2'] + df_dat['ask_price2'] * df_dat['bid_size2']) /\
                      (df_dat['bid_size2'] + df_dat['ask_size2'])
    fig, axes = plt.subplots(figsize=(32, 30), nrows=2)
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'bid_price1'], label='bid_price1', lw=2, color='tab:green')
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'ask_price1'], label='ask_price1', lw=2, color='tab:red')
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'bid_price2'], label='bid_price2', alpha=0.3, color='tab:green')
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'ask_price2'], label='ask_price2', alpha=0.3, color='tab:red')
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'wap1'], label='wap1', lw=2, linestyle='--', color='tab:blue')
    axes[0].plot(df_dat.loc[df_dat['time_id'] == time_id, 'wap2'], label='wap2', alpha=0.3, linestyle='--',  color='tab:blue')
    axes[1].plot(df_dat.loc[df_dat['time_id'] == time_id, 'bid_size1'], label='bid_size1', lw=2, color='tab:green')
    axes[1].plot(df_dat.loc[df_dat['time_id'] == time_id, 'ask_size1'], label='ask_size1', lw=2, color='tab:red')
    axes[1].plot(df_dat.loc[df_dat['time_id'] == time_id, 'bid_size2'], label='bid_size2', alpha=0.3, color='tab:green')
    axes[1].plot(df_dat.loc[df_dat['time_id'] == time_id, 'ask_size2'], label='ask_size2', alpha=0.3, color='tab:red')
    for i in range(2):
        axes[i].legend(prop={'size': 18})
        axes[i].tick_params(axis='x', labelsize=20, pad=10)
        axes[i].tick_params(axis='y', labelsize=20, pad=10)
    axes[0].set_ylabel('price', size=20, labelpad=15)
    axes[1].set_ylabel('size', size=20, labelpad=15)
    
    axes[0].set_title(
        f'Price {stock_id} time_id {time_id}',size=25,pad=15)
    axes[1].set_title(
        f'Sizes {stock_id} time_id {time_id}',size=25,pad=15)
    plt.show()

def feat_eda(df_dat,feat):
    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100, constrained_layout=True)
    t1 = 15
    l1 = 15
    sns.kdeplot(df_dat[feat], label='training set', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=l1)
    axes[0].tick_params(axis='y', labelsize=l1)
    axes[0].legend()
    axes[0].set_title(f'{feat} Dist', size=t1, pad=t1)
    sns.scatterplot(x=df_dat[feat], y=df_dat['target'], ax=axes[1])
    axes[1].set_title(f'{feat} vs target', size=t1, pad=t1)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', labelsize=l1)
    axes[1].tick_params(axis='y', labelsize=l1)
    plt.show()
    

def feat_dist(df_dat, feat):
    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100, constrained_layout=True)
    t1 = 15
    l1 = 15
    sns.kdeplot(df_dat[feat], label='training set', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=l1)
    axes[0].tick_params(axis='y', labelsize=l1)
    axes[0].legend()
    axes[0].set_title(f'{feat} Dist', size=t1, pad=t1)
    sns.scatterplot(x=df_dat[feat], y=df_dat['target'], ax=axes[1])
    axes[1].set_title(f'{feat} vs target', size=t1, pad=t1)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', labelsize=l1)
    axes[1].tick_params(axis='y', labelsize=l1)
    plt.show()