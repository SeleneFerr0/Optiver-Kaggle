#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:49:57 2021

@author: selene
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import pickle
import scipy as sc
import glob
import tqdm as tqdm
import seaborn as sns
from pandas.api.types import CategoricalDtype
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
import lightgbm as lgb
from xgboost import XGBRegressor
import xgboost as xgb


#%%
'''utility func'''
#wdd = r'/Users/irena/Documents/optiver'
wdd =  r'D:\optiver'
os.chdir(wdd)
global wd
wd = wdd
df_train = pd.read_csv(os.path.join(wdd, 'data', 'train.csv'))
cols = ['stock_id', 'time_id', 'target']

def met(y_t, y_p):
    return np.sqrt(np.mean(np.square((y_t - y_p) / y_t)))

def rmspe(y_t, y_p):  
    output = met(y_t, y_p)
    return 'rmspe', output, False


def group_recog(stock_groups, stock_id):
    for i in range(len(stock_groups)):
        if stock_id in stock_groups[i]:
            return i

def rv(slr):
    rev = np.sqrt(np.sum(slr**2))
    return rev


def wapvol(df):
    tmp = np.log(df).diff()
    v = np.sqrt(np.sum(tmp**2)) 
    return v


def log_return(st):
    return np.log(st).diff() 

def agg(df_st_book, df_stst, feature, func, new_name = None, rename = False):
    if rename:
        df_stst = pd.merge( df_st_book.groupby(by = ['time_id'])[feature].agg(func).reset_index().rename(columns = {feature : new_name}),
        df_stst, on = ['time_id'], how = 'left')    
    else:
        df_stst = pd.merge( df_st_book.groupby(by = ['time_id'])[feature].agg(func).reset_index(),
            df_stst, on = ['time_id'], how = 'left')     
    return df_stst


def inference1(models, stock_id, stock_groups, vol_feat, dg, grouped = True):
    if grouped:
        model = models[group_recog(stock_groups, stock_id)]
    else:
        model = models[stock_id]
    poly_feat = PolynomialFeatures(degree = dg)
    return model.predict(poly_feat.fit_transform([vol_feat]))[0]

def stock_trades(path, stock_id):
    trade =  pd.read_parquet(path)
    trade['stock_id'] = stock_id
    trade['trade_log_return'] = trade.groupby(['time_id'])['price'].apply(log_return).fillna(0)
    trade_features = ["price", "size", "order_count", "trade_log_return"]
    for feature in trade_features:
        if feature == "price":
            df_stst = agg(trade, trade, feature, func = "mean", rename = True, new_name = feature + "_mean")
        else:
            df_stst = agg(trade, df_stst, feature, func = "mean", rename = True, new_name = feature + "_mean")
        df_stst = agg(trade, df_stst, feature, func = max, rename = True, new_name = feature + "_max")
        df_stst = agg(trade, df_stst, feature, func = min, rename = True, new_name = feature + "_min")
        df_stst = agg(trade, df_stst, feature, func = sum, rename = True, new_name = feature + "_sum")
        if feature == "trade_log_return":
            df_stst = agg(trade, df_stst, feature, func = rv, rename = True, new_name = feature + "_rv")
    return df_stst

def load_trades(book):
    total_df = pd.DataFrame()
    for i in tqdm.tqdm(book):
        temp_stock = int(i.split("=")[1])
        temp_relvol = stock_trades(path = i, stock_id = temp_stock)
        total_df = pd.concat([total_df, temp_relvol])
    return total_df

def load_stock_info(path, stock_id):
    df_st_book = pd.read_parquet(path)
    #1st
    df_st_book['wap1'] = (df_st_book['bid_price1'] * df_st_book['ask_size1'] + df_st_book['ask_price1'] * df_st_book['bid_size1']) / (
                            df_st_book['bid_size1']+ df_st_book['ask_size1'])

    #2nd
    a = df_st_book['bid_price2'] * df_st_book['ask_size2'] + df_st_book['ask_price2'] * df_st_book['bid_size2']
    b = df_st_book['bid_size2']+ df_st_book['ask_size2']
    df_st_book['wap2'] = a/b
    
    #3rd
    a1 = df_st_book['bid_price1'] * df_st_book['ask_size1'] + df_st_book['ask_price1'] * df_st_book['bid_size1']
    a2 = df_st_book['bid_price2'] * df_st_book['ask_size2'] + df_st_book['ask_price2'] * df_st_book['bid_size2']
    b = df_st_book['bid_size1'] + df_st_book['ask_size1'] + df_st_book['bid_size2']+ df_st_book['ask_size2']    
    df_st_book['wap3'] = (a1 + a2)/ b
    
    #4th 
    a = (df_st_book['bid_price1'] * df_st_book['ask_size1'] + df_st_book['ask_price1'] * df_st_book['bid_size1']) / (df_st_book['bid_size1']+ df_st_book['ask_size1'])
    b = (df_st_book['bid_price2'] * df_st_book['ask_size2'] + df_st_book['ask_price2'] * df_st_book['bid_size2']) / (df_st_book['bid_size2']+ df_st_book['ask_size2'])
    df_st_book['wap4'] = (a + b) / 2
                    
    df_st_book['vol_wap1'] = (df_st_book.groupby(by = ['time_id'])['wap1'].apply(log_return).reset_index(drop = True).fillna(0))
    df_st_book['vol_wap_2'] = (df_st_book.groupby(by = ['time_id'])['wap2'].apply(log_return).reset_index(drop = True).fillna(0))
    df_st_book['vol_wap_3'] = (df_st_book.groupby(by = ['time_id'])['wap3'].apply(log_return).reset_index(drop = True).fillna(0))
    df_st_book['vol_wap_4'] = (df_st_book.groupby(by = ['time_id'])['wap4'].apply(log_return).reset_index(drop = True).fillna(0))
                
        
    df_st_book['bas'] = (df_st_book[['ask_price1', 'ask_price2']].min(axis = 1)
                                / df_st_book[['bid_price1', 'bid_price2']].max(axis = 1) - 1)   
    
    #spreads
    df_st_book['h_spread_l1'] = df_st_book['ask_price1'] - df_st_book['bid_price1']
    df_st_book['h_spread_l2'] = df_st_book['ask_price2'] - df_st_book['bid_price2']
    df_st_book['v_spread_b'] = df_st_book['bid_price1'] - df_st_book['bid_price2']
    df_st_book['v_spread_a'] = df_st_book['ask_price1'] - df_st_book['ask_price2']
    df_st_book['spread_dif1'] = df_st_book['ask_price1'] - df_st_book['bid_price2']
    df_st_book['spread_dif2'] = df_st_book['ask_price2'] - df_st_book['bid_price1']
    
    df_stst = pd.merge(
        df_st_book.groupby(by = ['time_id'])['vol_wap1'].agg(rv).reset_index(),
        df_st_book.groupby(by = ['time_id'], as_index = False)['bas'].mean(),
        on = ['time_id'], how = 'left'
    )
    
    vol_features = ["vol_wap_2", "vol_wap_3", "vol_wap_4"]
    sprd_feat = ["h_spread_l1", 'h_spread_l2', 'v_spread_b', 'v_spread_a', "spread_dif1", "spread_dif2"]
    for feature in vol_features:
         df_stst = agg(df_st_book, df_stst, feature, rv)
            
    for feature in sprd_feat:
        df_stst = agg(df_st_book, df_stst, feature, func = max, rename = True, new_name = feature + "_max")
        df_stst = agg(df_st_book, df_stst, feature, func = min, rename = True, new_name = feature + "_min")
        df_stst = agg(df_st_book, df_stst, feature, func = sum, rename = True, new_name = feature + "_sum")
        df_stst = agg(df_st_book, df_stst, feature, func = "mean", rename = True, new_name = feature + "_mean")
    
    df_stst['stock_id'] = stock_id
    return df_stst

def load_dat(bk):
    total_df = pd.DataFrame()
    for i in tqdm.tqdm(bk):
        tmp_stock = int(i.split("=")[1])
        tmp_relvol = load_stock_info(i, tmp_stock)
        total_df = pd.concat([total_df, tmp_relvol])
    return total_df


def xgb_lin_inf(X,y,dg):
    polyfeat = PolynomialFeatures(degree = dg)
    weights = 1/np.square(y)
    clf = XGBRegressor(eval_metric = rmspe, sample_weight = weights)
    x = np.array(X)
    x = np.array(X).reshape(-1,len(x[0]))
    X_ = polyfeat.fit_transform(x)
    res = clf.fit(X_, np.array(y).reshape(-1,1), sample_weight = weights)
    return res

def poly_transform(X, poly_feat):
    x = np.array(X)
    x = np.array(X).reshape(-1,len(x[0]))
    X = poly_feat.fit_transform(x)
    return X


def xgb_grids(X, y, degree = 1, params = {"reg_alpha" : 20, "reg_lambda" : 20, "max_depth" : 5, "n_estimators" : 500}, folds=10):
    polyfeat = PolynomialFeatures(degree = degree)
    skf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (tr_idx, ts_idx) in enumerate(skf.split(X)):
        
        x_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        x_ts, y_ts = X.iloc[ts_idx], y.iloc[ts_idx]
        
        x_tr = poly_transform(x_tr, polyfeat)
        x_ts = poly_transform(x_ts, polyfeat)
        weights = np.array(1/np.square(y_tr))
        eval_weights = np.array(1/np.square(y_ts))
        model = XGBRegressor(**params)
        model.fit(x_tr, y_tr,
                 eval_set=[(x_ts, y_ts)],
                 early_stopping_rounds=10, sample_weight = weights, sample_weight_eval_set = [eval_weights],
                  verbose=False)   
              
    return model

#Group
def gpst(gtt, n): 
    return [gtt[x: x+n] for x in range(0, len(gtt), n)]


def df_combine(train = True):
    if train:
        order_book_training = glob.glob('D:/optiver/data/book_train.parquet/*')
        total_df = load_dat(order_book_training)
        trade_training = glob.glob('D:/optiver/data/trade_train.parquet/*')
        total_trade = load_trades(trade_training).drop(columns = ["seconds_in_bucket", "price", "size", "order_count", "trade_log_return"]).drop_duplicates()
        total_df = total_df.merge(total_trade, on = ["stock_id","time_id"], how = "left")
    else:
        order_book_test = glob.glob('D:/optiver/data/book_test.parquet/*')
        test_df = load_dat(order_book_test)
        trade_test = glob.glob('/Users/irena/Documents/optiver/data/trade_test.parquet/*')
        trade_test = glob.glob('/Users/irena/Documents/optiver/data/trade_test.parquet/*')
        total_test = load_trades(trade_test).drop(columns = ["seconds_in_bucket", "price", "size", "order_count", "trade_log_return"]).drop_duplicates()
        dfres = test_df.merge(total_test, on = ["stock_id","time_id"], how = "left")
    return dfres


def fillna(mrg):
    nan_features = list(mrg.columns)[32:]
    missing_stock_ids = list(mrg[mrg["trade_log_return_rv"].isnull()]["stock_id"].unique())
    for stock_id in missing_stock_ids:
        for feature in nan_features:
            if mrg.loc[mrg["stock_id"] == stock_id, feature].isnull().all():
                mrg.loc[mrg["stock_id"] == stock_id, feature] = mrg.loc[mrg["stock_id"] == stock_id, feature].fillna(mrg[feature].mean())
            else:
                mrg.loc[mrg["stock_id"] == stock_id, feature] = mrg.loc[mrg["stock_id"] == stock_id, feature].fillna(mrg.loc[mrg["stock_id"] == stock_id, feature].mean())
    return mrg
#%%
df_mrged = pd.read_csv(os.path.join(wdd, 'data','df_all_bid_ask.csv'))
feat_lst = list(df_mrged.drop(columns = ["stock_id", "time_id", "target"]).columns)
df_mrged=fillna(df_mrged)

#%%
'''principle'''
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters=6, random_state=42)
#kmeans.fit(df_mrged[feat_lst])
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ss = StandardScaler()

X = df_mrged[feat_lst]
X_std = ss.fit_transform(X)


cov_mat = np.cov(X_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

pca = PCA(n_components=3) 
X_pca = pca.fit_transform(X_std)

from mpl_toolkits.mplot3d import Axes3D


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
'''show pca'''
fig, ax = plt.subplots(figsize=(32, 10))
ax.scatter(st_pca[:, 0], st_pca[:, 1], s=200, c=kmeans.labels_, cmap='RdBu')
for idx, stock_id in enumerate(df_mrged['stock_id'].values):
    ax.annotate(stock_id, (st_pca[idx, 0], st_pca[idx, 1]), fontsize=20)
    
ax.tick_params(axis='x', labelsize=20, pad=10)
ax.tick_params(axis='y', labelsize=20, pad=10)
ax.set_title('', size=25, pad=20)

plt.show()


#%%
'''check'''
df_all = fillna(df_mrged)
nachk1 = df_all[df_all["order_count_max"].isnull()]

assert len(nachk1)==0

fig, ax = plt.subplots(figsize=(25,25))
sns.heatmap(df_all.groupby(by = "stock_id").mean()[feat_lst + ["target"]].corr(), center = 0, annot = True, cmap="YlGnBu", linewidths = .05)
plt.title("Correlation Matrix")
df_all.groupby("stock_id").mean()["target"].reset_index().sort_values(by = "target").plot(x = "stock_id", y = "target", kind = "bar", figsize = (20, 8), title = "Mean tgt vol.")

#%%

'''features'''

order_book_test = glob.glob('D:/optiver/data/book_test.parquet/*')
test_df = load_dat(order_book_test)


order_trade_test = glob.glob('D:/optiver/data/trade_test.parquet/*')
total_test = load_trades(order_trade_test)

cols = ['time_id', 'trade_log_return_rv', 'trade_log_return_sum', 'trade_log_return_min', 'trade_log_return_max', 'trade_log_return_mean', 'order_count_sum', 'order_count_min', 'order_count_max', 'order_count_mean', 'size_sum', 'size_min', 'size_max', 'size_mean', 'price_sum', 'price_min', 'price_max', 'price_mean', 'seconds_in_bucket', 'price', 'size', 'order_count', 'stock_id', 'trade_log_return']

total_test = total_test.drop(columns = ["seconds_in_bucket", "price", "size", "order_count", "trade_log_return"]).drop_duplicates()
total_df = test_df.merge(total_test, on = ["stock_id","time_id"], how = "left")

# df_test = pd.read_csv(os.path.join(wdd, 'data','test.csv'))

df_test= total_df.copy()

nas = df_test[df_test["order_count_max"].isnull()]

df_test = fillna(df_test)
np.any(np.isnan(df_test))
np.all(np.isfinite(df_test))


#%%
'''LGBM baseline'''

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def rmspe_f(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

def lgbm_train(train, test):
    x = train.drop(['target', 'time_id'], axis = 1)
    y = train['target']
    # y_test = test['target']
    x_test = test.drop(['target','time_id'], axis = 1)
    x['stock_id'] = x['stock_id'].astype(int)
    x_test['stock_id'] = x_test['stock_id'].astype(int)
    params = {
      'objective': 'rmse',  
      'boosting_type': 'gbdt',
      'num_leaves': 50,
      'n_jobs': -1,
      'learning_rate': 0.1,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'verbose': -1
    }
    pred_oof = np.zeros(x.shape[0])
    pred_test = np.zeros(x_test.shape[0])
    kfold = KFold(n_splits = 5, random_state = 66, shuffle = True)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights, categorical_feature = ['stock_id'])
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights, categorical_feature = ['stock_id'])
        model = lgb.train(params = params, train_set = train_dataset, valid_sets = [train_dataset, val_dataset], 
                          num_boost_round = 10000, early_stopping_rounds = 50, verbose_eval = 50,feval = rmspe_f)
        
        pred_oof[val_ind] = model.predict(x_val)
        pred_test += model.predict(x_test) / 5
        
    rmspe_score = rmspe(y, pred_oof)
    print(f'out of folds RMSPE is {rmspe_score}')
    return pred_test



stid_lst = df_all['stock_id'].unique()
t_lst = df_all['time_id'].unique()


train_ids = stid_lst[:88]
test_ids = stid_lst[88:]


train_ids = t_lst[:3060]
test_ids = t_lst[3060:]


df_lg_train = df_all[df_all.time_id.isin(train_ids)]
df_lg_test = df_all[df_all.time_id.isin(test_ids)]
lg_res = lgbm_train(df_lg_train, df_lg_test)

df_lg_test['pred'] = lg_res 
df_lg_test['res_dif'] = (df_lg_test['target']-df_lg_test['pred'])/df_lg_test['target']
np.sqrt(np.mean(np.square(df_lg_test['res_dif'])))

#df_lg_test.to_csv("D:/optiver/lgbm_baseline_test.csv")


lg_tr_res = lgbm_train(df_lg_train, df_lg_train)
df_lg_train['pred'] = lg_tr_res
df_lg_train['res_dif'] = (df_lg_train['target']-df_lg_train['pred'])/df_lg_train['target']
np.sqrt(np.mean(np.square(df_lg_train['res_dif'])))


#df_lg_train.to_csv("D:/optiver/lgbm_baseline_train.csv")

plt.figure(figsize=(20,5))
plt.xlabel('lgbm predicion in 10 minutes window')

ax1 = df_lg_test.pred.plot(color='blue', grid=True, label='lgb-pred',alpha=0.9)
ax2 = df_lg_test.target.plot(color='red', grid=True, secondary_y=True, label='target',alpha=0.5)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()


plt.legend(h1+h2, l1+l2, loc=2)
plt.show()

#%%
'''test run'''
stock_id_train = df_train.stock_id.unique()
ln_models = {}
degree = 1
coefs = {}


def group_recog(sg, stock_id):
    for i in range(len(sg)):
        if stock_id in sg[i]:
            return i


def inference1(models, stock_id, stock_groups, vol_feat, degree, grouped = True):
    if grouped:
        model = models[group_recog(stock_groups, stock_id)]
    else:
        model = models[stock_id]
    polyfeat = PolynomialFeatures(degree = degree)
    return model.predict(polyfeat.fit_transform([vol_feat]))[0]


#order_book_test = glob.glob('D:/optiver/data/book_test.parquet/*')
#df_test = load_dat(order_book_test)
#train_join = df_all[df_all.stock_id.isin(stock_id_train)]

def train(df_train, df_combined, feature_list, group_number = 5, group = True, degree = 1, xgmethod = False):
    stock_groups = gpst(list(df_combined.groupby("stock_id").mean()["target"].reset_index().sort_values(by = "target")["stock_id"]), group_number)
    stock_id_train = df_train.stock_id.unique() 
    models = {}
    if not group:
        for i in tqdm.tqdm(stock_id_train):
            temp = df_combined[df_combined["stock_id"]==i]
            X = temp[feature_list]
            y = temp["target"]
            if xgmethod:
                models[i] = xgb_grids(X, y, degree = degree, folds=5)
            else:
                models[i] = xgb_lin_inf(X,y,degree)
    else:
        for i in tqdm.tqdm(range(len(stock_groups))):
            temp = df_combined[df_combined["stock_id"].isin(stock_groups[i])]
            X = temp[feature_list]
            y = temp["target"]
            if xgmethod:
                models[i] = xgb_grids(X, y, degree = degree, folds=5)
            else:
                models[i] = xgb_lin_inf(X,y,degree)
    return models



def inference1(models, stock_id, stock_groups, vol_feat, degree, grouped = True):
    if grouped:
        model = models[group_recog(stock_groups, stock_id)]
    else:
        model = models[stock_id]
    polyfeat = PolynomialFeatures(degree = degree)
    return model.predict(polyfeat.fit_transform([vol_feat]))[0]


def subm(test_df, df_combined, models, stock_groups, feature_list, degree = 1, merge = False):
    
    
    submission = pd.DataFrame({"row_id" : [], "target" : []})  
    submission["row_id"] = test_df.apply(lambda x: str(int(x.stock_id)) + '-' + str(int(x.time_id)), axis=1)
    submission["target"] = test_df.apply(lambda x: inference1(models, x.stock_id,stock_groups, list(x[feat_lst]),degree),axis = 1)
    
    if merge:
        submission["stock_id"] = test_df.apply(lambda x: int(x.stock_id), axis = 1)
        submission["time_id"] = test_df.apply(lambda x: int(x.time_id), axis=1)
        overall = df_combined.merge(submission, on = ["stock_id", "time_id"], how = "left")
        return overall
    else:
        return submission     

def train_score(df_combined, models, stock_groups, feature_list, degree = 1):
    train_pred = subm(df_combined, df_combined, models, stock_groups, feature_list, degree = degree, merge = True)
    rmspe_train = rmspe(np.array(train_pred["target_x"]), np.array(train_pred["target_y"]))
    return rmspe_train
    
#%%
now = datetime.now() 
dtime = now.strftime("%m-%d-%Y-%H-%M")
group_numbers = [4]
degrees = [2]
scores = {}
vald = True

'''xgb'''

#stock_id_train = df_train.stock_id.unique()
#stock_id_train = stock_id_train[:-24]
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#


df_test2 = clean_dataset(df_test)

for group_number in group_numbers:
    for degree in degrees:
        stock_groups = gpst(list(df_all.groupby("stock_id").mean()["target"].reset_index().sort_values(by = "target")["stock_id"]), group_number)
        models = train(df_train, df_all,feat_lst, group_number = group_number, group = True, degree = degree, xgmethod = True)
        if vald:
            overall_train_score = train_score(df_all, models, stock_groups, feat_lst, degree = degree)
            print("score for gp # {}, dg {}:".format(group_number, degree), overall_train_score)
            scores[str(group_number) + " - " + str(degree)] = overall_train_score


#stock_id_test = df_train.stock_id.unique()
#stock_id_test = stock_id_train[-24:]
#df_test = df_all[df_all.stock_id.isin(stock_id_test)]

xgb_sub = subm(df_test2, df_all, models, stock_groups, feat_lst, degree = 2, merge = False)
xgb_tr_sub = subm(df_all, df_all, models, stock_groups, feat_lst, degree = 2, merge = False)

#%%    
df_xg_all = df_all[df_all.time_id.isin(train_ids)]
df_xg_test = df_test2[df_test2.time_id.isin(test_ids)]

df_xg_train = df_train[df_train.time_id.isin(train_ids)]

now = datetime.now() 
dtime = now.strftime("%m-%d-%Y-%H-%M")
group_numbers = [4]
degrees = [2]
scores = {}
vald = True

#%%
for group_number in group_numbers:
    for degree in degrees:
        stock_groups = gpst(list(df_xg_all.groupby("stock_id").mean()["target"].reset_index().sort_values(by = "target")["stock_id"]), group_number)
        models = train(df_xg_train, df_xg_all,feat_lst, group_number = group_number, group = True, degree = degree, xgmethod = True)
        if vald:
            overall_train_score = train_score(df_xg_all, models, stock_groups, feat_lst, degree = degree)
            print("score for gp # {}, dg {}:".format(group_number, degree), overall_train_score)
            scores[str(group_number) + " - " + str(degree)] = overall_train_score

#%%
xgb_sub2 = subm(df_test2, df_all, models, stock_groups, feat_lst, degree = 2, merge = False)
xgb_tr_sub2 = subm(df_all, df_all, models, stock_groups, feat_lst, degree = 2, merge = False)

#%%
xgb_sub2.to_csv(str(dtime) + "_xgb_subdg2.csv", index = False)
xgb_tr_sub2.to_csv(str(dtime) + "_xgb_tr_subdg2.csv", index = False)
#%%  

xgb_sub.to_csv(str(dtime) + "_xgb_subdg1.csv", index = False)
xgb_tr_sub.to_csv(str(dtime) + "_xgb_tr_subdg1.csv", index = False)

all_tr_score = train_score(df_all, models, stock_groups, feat_lst, degree = degree)
print("Train score for group number {} and degree {}:".format(group_number, degree), all_tr_score)
scores[str(group_number) + " - " + str(degree)] = all_tr_score 
#all_tr_score.to_csv(str(dtime) + "_xgb_score.csv", index = False)


#%%

xgb_results = pd.read_csv(os.path.join(wdd,'output', '1129', '11-29-2021-23-50_xgb_sub.csv'))
xgb_results.rename(columns={'target':'xbg_pred'},inplace=True)

df_lg_test['row_id'] = df_lg_test['stock_id'].astype(str)  +'-' + df_lg_test['time_id'].astype(str) 
compr= pd.merge(df_lg_test, xgb_sub, on='row_id', how = 'left')
compr = compr[~compr["xbg_pred"].isnull()]


xgb_tr_sub.rename(columns={'target':'xbg_pred'},inplace=True)
df_lg_train['row_id'] = df_lg_train['stock_id'].astype(str)  +'-' + df_lg_train['time_id'].astype(str) 
compr2= pd.merge(df_lg_train, xgb_tr_sub, on='row_id', how = 'left')


comp2r = compr2[~compr2["xbg_pred"].isnull()]


res_all = df_lg_train.append(df_lg_test)
ck2 = res_all[res_all.row_id.isnull()]

compr2= pd.merge(res_all, xgb_tr_sub, on='row_id', how = 'left')
ck3 = compr2[compr2.pred.isnull()]

compr2.to_csv(str(dtime) + "_xgb_all.csv", index = False)

t = xgb_results[xgb_results.row_id.str.contains('103-9')==True]

compr['xbg_dif'] = (compr['target']-compr['xbg_pred'])/compr['target']

a = np.sqrt(np.mean(np.square(compr['res_dif'])))
b = np.sqrt(np.mean(np.square(compr['xbg_dif'])))

#compr.to_csv("test_Results.csv", index = False)

compr[compr.stock_id == 100 ].plot(x="time_id", y=["target", "pred"], kind="line")
compr[compr.stock_id == 100 ].plot(x="time_id", y=["target", "xbg_pred"], kind="line")



#%%
compr3 = df_all.copy()
compr3['row_id'] = compr3['stock_id'].astype(str)  +'-' + compr3['time_id'].astype(str) 
compr3= pd.merge(compr3, xgb_tr_sub2, on='row_id', how = 'left')
ck3 = compr3[compr3.target_y.isnull()]

compr3.rename(columns={'target_x':'target', 'target_y':'xgb_pred2'},inplace=True)

compr3.to_csv(str(dtime) + "res_subdg2.csv", index = False)