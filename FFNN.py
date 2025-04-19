from PreprocessFFNN import *

corr = train_p.corr()
ids = corr.index
kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
labels = kmeans.labels_
l = []
for n in range(7):
    l.append ( [ (x-1) for x in ( (ids+1)*(labels == n)) if x > 0] )
    
matrice = []
matriceTest = []
n = 0
for ind in l:
    ffnn_w_corr = train.loc[train['stock_id'].isin(ind) ]
    ffnn_w_corr = ffnn_w_corr.groupby(['time_id']).agg(np.nanmean)
    ffnn_w_corr.loc[:,'stock_id'] = str(n)+'c1'
    matrice.append ( ffnn_w_corr )
    
    ffnn_w_corr = test.loc[test['stock_id'].isin(ind) ]    
    ffnn_w_corr = ffnn_w_corr.groupby(['time_id']).agg(np.nanmean)
    ffnn_w_corr.loc[:,'stock_id'] = str(n)+'c1'
    matriceTest.append ( ffnn_w_corr )
    
    n+=1
    
matrice1 = pd.concat(matrice).reset_index()
matrice1.drop(columns=['target'],inplace=True)

matrice2 = pd.concat(matriceTest).reset_index()

matriceTest = []
matrice = []
kmeans = []
matrice2 = pd.concat([matrice2,matrice1.loc[matrice1.time_id==5]])
matrice1 = matrice1.pivot(index='time_id', columns='stock_id')
matrice1.columns = ["_".join(x) for x in matrice1.columns.ravel()]
matrice1.reset_index(inplace=True)

matrice2 = matrice2.pivot(index='time_id', columns='stock_id')
matrice2.columns = ["_".join(x) for x in matrice2.columns.ravel()]
matrice2.reset_index(inplace=True)

nnn = ['time_id',
     'log_return1_realized_volatility_0c1',
     'log_return1_realized_volatility_1c1',     
     'log_return1_realized_volatility_3c1',
     'log_return1_realized_volatility_4c1',     
     'log_return1_realized_volatility_6c1',
     'total_volume_mean_0c1',
     'total_volume_mean_1c1', 
     'total_volume_mean_3c1',
     'total_volume_mean_4c1', 
     'total_volume_mean_6c1',
     'trade_size_mean_0c1',
     'trade_size_mean_1c1', 
     'trade_size_mean_3c1',
     'trade_size_mean_4c1', 
     'trade_size_mean_6c1',
     'trade_order_count_mean_0c1',
     'trade_order_count_mean_1c1',
     'trade_order_count_mean_3c1',
     'trade_order_count_mean_4c1',
     'trade_order_count_mean_6c1',      
     'price_spread_mean_0c1',
     'price_spread_mean_1c1',
     'price_spread_mean_3c1',
     'price_spread_mean_4c1',
     'price_spread_mean_6c1',   
     'bid_spread_mean_0c1',
     'bid_spread_mean_1c1',
     'bid_spread_mean_3c1',
     'bid_spread_mean_4c1',
     'bid_spread_mean_6c1',       
     'ask_spread_mean_0c1',
     'ask_spread_mean_1c1',
     'ask_spread_mean_3c1',
     'ask_spread_mean_4c1',
     'ask_spread_mean_6c1',   
     'volume_imbalance_mean_0c1',
     'volume_imbalance_mean_1c1',
     'volume_imbalance_mean_3c1',
     'volume_imbalance_mean_4c1',
     'volume_imbalance_mean_6c1',       
     'bid_ask_spread_mean_0c1',
     'bid_ask_spread_mean_1c1',
     'bid_ask_spread_mean_3c1',
     'bid_ask_spread_mean_4c1',
     'bid_ask_spread_mean_6c1',
     'size_sum2_0c1',
     'size_sum2_1c1',
     'size_sum2_3c1',
     'size_sum2_4c1',
     'size_sum2_6c1'] 

train = pd.merge(train,matrice1[nnn],how='left',on='time_id')
test = pd.merge(test,matrice2[nnn],how='left',on='time_id')
matrice1 = []
matrice2 = []

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

hidden_units = (128,64,32)
stock_embedding_size = 24

stocks_col = train['stock_id']
def mod():
    id_input = keras.Input(shape=(1,), name='stock_id')
    num_input = keras.Input(shape=(362,), name='num_data')
    stock_embedded = keras.layers.Embedding(max(stocks_col)+1, stock_embedding_size, 
                                           input_length=1, name='stock_embedding')(id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='swish')(out)
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)
    
    model = keras.Model(
    inputs = [id_input, num_input],
    outputs = out,
    )
    
    return model

model_nn = 'NN'
pred_name = 'pred_{}'.format(model_nn)

n_folds = 5
kf = model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=2020)
err_folds[model_nn] = []
counter = 1

features_to_consider = list(train)

features_to_consider.remove('time_id')
features_to_consider.remove('target')
features_to_consider.remove('row_id')
try:
    features_to_consider.remove('pred_NN')
except:
    pass


train[features_to_consider] = train[features_to_consider].fillna(train[features_to_consider].mean())
test[features_to_consider] = test[features_to_consider].fillna(train[features_to_consider].mean())

train[pred_name] = 0
test['target'] = 0

for n_count in range(n_folds):
    print('CV {}/{}'.format(counter, n_folds))
    
    indexes = np.arange(n_folds).astype(int)    
    indexes = np.delete(indexes,obj=n_count, axis=0) 
    
    indexes = np.r_[values[indexes[0]],values[indexes[1]],values[indexes[2]],values[indexes[3]]]
    
    X_train = train.loc[train.time_id.isin(indexes), features_to_consider]
    y_train = train.loc[train.time_id.isin(indexes), target_name]
    X_test = train.loc[train.time_id.isin(values[n_count]), features_to_consider]
    y_test = train.loc[train.time_id.isin(values[n_count]), target_name]
    model = mod()
    
    model.compile(
        keras.optimizers.Adam(learning_rate=0.005),
        loss=root_mean_squared_per_error
    )
    
    try:
        features_to_consider.remove('stock_id')
    except:
        pass
    
    num_data = X_train[features_to_consider]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))         
    num_data = scaler.fit_transform(num_data.values)    
    
    stocks_col = X_train['stock_id']    
    target =  y_train
    
    num_data_test = X_test[features_to_consider]
    num_data_test = scaler.transform(num_data_test.values)
    stocks_col_test = X_test['stock_id']

    model.fit([stocks_col, num_data], target, batch_size=1024,epochs=500, validation_data=([stocks_col_test, num_data_test], y_test), 
            callbacks=[es, plateau],validation_batch_size=len(y_test),shuffle=True,verbose = 1)

    preds = model.predict([stocks_col_test, num_data_test]).reshape(1,-1)[0]
    
    err = round(rmspe(y_true = y_test, y_pred = preds),5)
    err_folds[model_nn].append(err)
    
    tt =scaler.transform(test[features_to_consider].values)
    test[target_name] += model.predict([test['stock_id'], tt]).reshape(1,-1)[0].clip(0,1e10)
       
    counter += 1
    features_to_consider.append('stock_id')

test[target_name] = test[target_name]/n_folds
err = round(rmspe(y_true = train[target_name].values, y_pred = train[pred_name].values),5)
test[['row_id', target_name]].to_csv('test1.csv',index = False)
test1 = pd.read_csv("test1.csv")
test1 = test1.rename(columns={'row_id': 'row_id', 'target': 'target_pred'})
testtrue = pd.read_csv("newtest.csv")
pred = test1["target_pred"]
test3 = testtrue.join(pred)
test3["error"] = rmspe(test3["target"], test3["target_pred"])