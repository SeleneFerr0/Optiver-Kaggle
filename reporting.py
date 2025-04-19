# -*- coding: utf-8 -*-
"""
Created on Mon Nov3 05:50:25 2021

@author: selene

reporting

"""

import numpy as np
import os
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from pandas.api.types import CategoricalDtype
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import zipfile
import plotly.graph_objects as go
import pandas as pd
from utils import *
from google_drive_downloader import GoogleDriveDownloader as gdd

import plotly.io as pio
#wdd = r'/Users/irena/Documents/optiver'
wdd =  r'D:\optiver'


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


os.chdir(wdd)

df_train = pd.read_csv(os.path.join(wdd, 'data', 'train.csv'))
cols = ['stock_id', 'time_id', 'target']

df_test = pd.read_csv(os.path.join(wdd, 'data', 'test.csv'))
cols = ['stock_id', 'time_id', 'target']
# 126 stocks
#cnt_var(raw,'stock_id')



#%%
#pio.renderers.default='svg'
pio.renderers.default = 'browser'
if os.path.isfile('./xgb_down.zip'):
    pass
else:
    gdd.download_file_from_google_drive(file_id='1YnqKDKirb-cmNW-eeUh3cHp750TQl1bm',
                                        dest_path='./xgb_down.zip',
                                        unzip=True)


zf = zipfile.ZipFile('./xgb_down.zip')
df = pd. read_csv(zf. open('12-06-2021-10-51_xgb_all.csv'))
#df= pd.read_csv(('./xgb_down.zip'))
df['stock_id'] = df['stock_id'].astype(str)

DEFAULT_TICKERS = list(df.stock_id.unique())

#%%
stid = '18'
stid = '43'


tmp = df[df.stock_id == stid]
f = plt.figure(figsize=(16, 8))
gs = f.add_gridspec(1, 2)

with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0, 0])
    tmp['trade_log_return_mean'].hist(bins = 100,color='tomato', range=[-0.001,0.001])
    plt.title("Return distribution")
    
with sns.axes_style("whitegrid"):
    ax = f.add_subplot(gs[0, 1])
    tmp['spread_dif1_mean'].hist(bins = 60,color='orange', range=[0,0.006])
    plt.title("spread")



#%%
## feature distribution
# correlation

for stock_id in tqdm(sorted(df_train['stock_id'].unique())):
    df_dat = df_raw_dataload('train', stock_id)
    df_dat['bid_ask_price_ratio'] = df_dat['bid_price1'] / df_dat['ask_price1']
    for agg in ['mean', 'std', 'min', 'max', realized_volatility]:
        bid_ask_price_ratio_aggregation = df_dat.groupby('time_id')['bid_ask_price_ratio'].agg(agg)
        feature_name = agg.__name__ if callable(agg) else agg
        df_train.loc[df_train['stock_id'] == stock_id, f'book_bid_ask_price_ratio_{feature_name}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(bid_ask_price_ratio_aggregation)
        
    df_dat['wap1'] = (df_dat['bid_price1'] * df_dat['ask_size1'] + df_dat['ask_price1'] * df_dat['bid_size1']) /\
                      (df_dat['bid_size1'] + df_dat['ask_size1'])
    df_dat['wap2'] = (df_dat['bid_price2'] * df_dat['ask_size2'] + df_dat['ask_price2'] * df_dat['bid_size2']) /\
                      (df_dat['bid_size2'] + df_dat['ask_size2'])

    for wap in [1, 2]:
        for agg in ['mean', 'std', 'min', 'max', realized_volatility]:
            wap_aggregation = df_dat.groupby('time_id')[f'wap{wap}'].agg(agg)
            feature_name = agg.__name__ if callable(agg) else agg
            df_train.loc[df_train['stock_id'] == stock_id, f'wap{wap}_{feature_name}'] = df_train[df_train['stock_id'] == stock_id]['time_id'].map(wap_aggregation)


for feat in df_train.columns[6:]:
    feat_dist(feat)


plot_corr(df_train.iloc[:, 2:])

tmp = df[df.stock_id == '125']
fig = go.Figure(data=[go.Candlestick(x=tmp['time_id'],
                open=tmp['price_mean'],
                high=tmp['price_max'],
                low=tmp['price_min'],
                close=tmp['price_mean'])])
    
fig.show()




fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Candlestick(x=df['time_id'],
                open=tmp['trade_log_return_mean'], high=tmp['trade_log_return_max'],
                low=tmp['trade_log_return_min'], close=tmp['trade_log_return_mean']),
               secondary_y=True)

fig.add_trace(go.Bar(x=tmp['time_id'], y=tmp['size_mean']),
               secondary_y=False)

fig.layout.yaxis2.showgrid=False
fig.show()


#%%
X = np.array(df_train.iloc[:, 1:])
y = df_train['target']

ss = StandardScaler()


X_std = ss.fit_transform(X)


cov_mat = np.cov(X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


xs = X_pca.T[0]
ys = X_pca.T[1]
zs = X_pca.T[2]



ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plot = ax.scatter(xs, ys, zs, alpha=0.75,
                  c=df_mrged['target'], cmap='viridis', depthshade=True)


fig.colorbar(plot, shrink=0.6, aspect=9)
plt.show()


var_imp = [np.abs(pca.components_[i]).argmax() for i in range(3)]
var_imp_all = pd.DataFrame(pca.components_, columns=X.columns)
var_imp_all


initial_feature_names = X.columns
var_names = [initial_feature_names[var_imp[i]] for i in range(3)]
dic = {'PC{}'.format(i): var_names[i] for i in range(3)}
var_pca = pd.DataFrame(dic.items())
var_pca


#%%
plt.figure(figsize=(20,5))
plt.xlabel('lgboost predicion in 10 minutes window')

tmp = df[df.time_id>= 25653]

ax1 = tmp.pred.plot(color='blue', grid=True, label='lgb-pred',alpha=0.9)
ax2 = tmp.target.plot(color='red', grid=True, secondary_y=True, label='target',alpha=0.5)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()


plt.legend(h1+h2, l1+l2, loc=2)
plt.show()