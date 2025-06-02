import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
import mlflow
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LSTM Model for Gold Price Prediction")

df = pd.read_csv('preprocessing/processed_gold_price.csv', parse_dates=['Date'])

test_size = df[df.Date.dt.year==2022].shape[0]
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1,1))

window_size = 60

train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1,1))

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
test_data = df.Price[-test_size-60:]
test_data = scaler.transform(test_data.values.reshape(-1,1))

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
    
    
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_train = np.reshape(y_train, (-1,1))
y_test  = np.reshape(y_test, (-1,1))

def define_model(units, dropout_rate):
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = units, return_sequences=True)(input1)  
    x = Dropout(dropout_rate)(x)
    x = LSTM(units = units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units = units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()
    return model
with mlflow.start_run():
    dropout_rate = 0.2
    units = 64
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("epochs", 150)
    mlflow.log_param("batch_size", 32)
    model = define_model(units, dropout_rate)
    mlflow.autolog()
    mlflow.keras.log_model(
        model,
        artifact_path="model",
        registered_model_name="LSTM_Gold_Price_Prediction_Model",
    )
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
    y_pred = model.predict(X_test)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    Accuracy = 1 - MAPE
    mlflow.log_metric("MAPE", float(MAPE))
    mlflow.log_metric("Accuracy", float(Accuracy))
