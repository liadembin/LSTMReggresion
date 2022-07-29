import pandas as pd 
import numpy as np 
from pandas_profiling import ProfileReport
normalized_data=pd.read_csv("https://raw.githubusercontent.com/liadembin/LSTMReggresion/main/data/normalized.csv")
print("read_csv")
normalized_data=normalized_data[normalized_data.columns[1:]]
normalized_data.head(3)
normalized_data['Adj Close'].plot()
window_size = 60
trainTestSplit=.9
data_arr = normalized_data.to_numpy()
X = []
y = []
for i in range(0,len(data_arr)-window_size):
    X.append(data_arr[i:i+window_size])
    y.append(data_arr[i+window_size,0])
X = np.array(X)
y = np.array(y)
print("X shape: ",X.shape)
print("y shape: ",y.shape)
index = int(len(X) * .9)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout,GRU,Flatten
model = Sequential()
model.add(LSTM(256,input_shape=X_train[0].shape,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(64,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()
import tensorflow.keras.backend as K
import tensorflow as tf
def smape_loss(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape * 100.0
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae','mse','mean_absolute_percentage_error',smape_loss])
print("compiled")
# with tf.device('/device:GPU:0'):
if True:
    history = model.fit(X_train,y_train,epochs=500
                      ,batch_size=128,shuffle=True,use_multiprocessing=True,validation_data=(X_test,y_test))
    df = pd.DataFrame(history.history)
    model.save("./newModel28_7_22/modelFinal.h5")
    df.to_csv("./newModel28_7_22/metrics.csv")
    # df.to_csv("/content/gdrive/My Drive/metrics.csv")
    # model.save("/content/gdrive/My Drive/NewLSTModelNormalizedDataStackedLstm27_7_22_NIGHT_LAST_ATTEMPT.h5")

model.save("./models/superStackLSTM.h5")

model.evaluate(X_test,y_test)

model.predict(X[index:])[-1,0] * max(data['Adj Close'])

X[index:].shape

max(data['Adj Close'])

df = pd.DataFrame({
    "real":y[0:],
    "predicted":model.predict(X[0:])[0:,0]
})
df.plot()

model.save('./models/LastNightStackedLSTM.h5')

old_model = load_model("./models/goodModelDividedByMaxAdjClose.h5",custom_objects={"smape_loss":smape_loss})

cols = ['Adj Close', '40 period ADX.', 'AO', 'UPPER', 'LOWER',
       '40 period ATR', 'Buy.', 'Sell.', 'Buy..1', 'Sell..1', 'BB_UPPER',
       'BB_MIDDLE', 'BB_LOWER', '40 period BBWITH', 'Balance Of Power',
       'Short.', 'Long.', 'CMO', 'Coppock Curve', '40 period DEMA', 'DI+',
       'DI-', 'LOWER.1', 'MIDDLE', 'UPPER.1', 'Bull.', 'Bear.',
       '40 period EMA', '40 period ER', '10 period EVSTC',
       '40 period EVWMA.', 'MACD', 'SIGNAL', '40 period FISH.',
       '40 period FRAMA.', '0', '40 period HMA.', 'TENKAN', 'KIJUN',
       'CHIKOU', 'IFT_RSI', '40 period KAMA.', 'KC_UPPER', 'KC_LOWER',
       'MACD.1', 'SIGNAL.1', '40 period MFI', 'Mass Index', 'BB_UPPER.1',
       'BB_MIDDLE.1', 'BB_LOWER.1', 'MOM', 'MSD', '%b', 'pivot', 's1',
       's2', 's3', 's4', 'r1', 'r2', 'r3', 'r4', 'pivot.1', 's1.1',
       's2.1', 's3.1', 's4.1', 'r1.1', 'r2.1', 'r3.1', 'r4.1', 'PPO',
       'SIGNAL.2', 'HISTO', 'psar', '40 period PZO', 'ROC',
       '40 period RSI', '0.1', '40 period SMA', '40 period SMM', 'SMMA',
       '40 period SQZMI', '40 period SSMA', '10 period STC',
       '40 period STOCH %K', '14 period stochastic RSI.',
       '40 period TEMA', 'TP', 'TR', '40 period TRIX', 'TSI', '0.2',
       'VBM', 'VFI', 'VIm', 'VIp', 'VWAP.', 'MACD.2', 'SIGNAL.3', 'VZO',
       '40 Williams %R', '40 period WMA.', 'WT1.', 'WT2.',
       '40 period ZLEMA', 'pct', 'pct_shifted']
for i in range(len(cols)):
    cols[i] = cols[i].replace("40","60")

data_old = data[cols]

data_old.head(3)

old_model.predict(X[index:])

