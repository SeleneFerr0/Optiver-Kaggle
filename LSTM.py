from PreprocessLSTM import *

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, Y_train = X.loc[train_index], y[train_index]
    X_test, Y_test = X.loc[test_index], y[test_index]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
n = 30  
Xtrain = []
Ytrain = []
for i in range(n, len(X_train_scaled)): 
    Xtrain.append(X_train_scaled[(i-n):i, :X_train_scaled.shape[1]])
    Ytrain.append(Y_train[i-1:i])
Xtrain, Ytrain = (np.array(Xtrain), np.array(Ytrain))
    
Xtest = []
Ytest = []
m = 30
for i in range(m, len(X_test)): 
    Xtest.append(X_test_scaled[(i-m):i, :X_test_scaled.shape[1]])
    Ytest.append(Y_test[i-1:i]) # predict the next record
Xtest, Ytest = (np.array(Xtest), np.array(Ytest))

def generate_layer(n_layers, n_nodes, activation, drop=None, d_rate=.5):
    for x in range(1,n_layers+1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass

model = Sequential()
model.add(LSTM(50, activation="tanh", return_sequences=True, input_shape=(30, Xtrain.shape[1])))
generate_layer(n_layers=2,n_nodes=30, activation="tanh",drop=1,d_rate=.1)
model.add(LSTM(30, activation="tanh"))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
LSTM_Model = model.fit(Xtrain, Ytrain, epochs = 100, batch_size =32, validation_split=0.1)
y_pred = model.predict(Xtest)
RMSPE = round(rmspe(y_true = Ytest, y_pred = y_pred),3)

