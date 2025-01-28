#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data = pd.read_csv('AAPL.csv')
print(data.head())


# In[31]:


print(data.tail())


# In[7]:


data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]


# In[8]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])


# In[9]:


import numpy as np
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data['Close'].values, seq_length)


# In[10]:


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))


# In[12]:


lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)


# In[13]:


data['Lag_1'] = data['Close'].shift(1)
data['Lag_2'] = data['Close'].shift(2)
data['Lag_3'] = data['Close'].shift(3)
data = data.dropna()


# In[14]:


X_lin = data[['Lag_1', 'Lag_2', 'Lag_3']]
y_lin = data['Close']
X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]


# In[15]:


from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)


# In[16]:


X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)


# In[17]:


lin_predictions = lin_model.predict(X_test_lin)
lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))


# In[25]:


hybrid_predictions = (0.7 * lstm_predictions) + (0.5 * lin_predictions)


# In[23]:


print(lstm_predictions.shape)
print(lin_predictions.shape)


# In[24]:


min_length = min(len(lstm_predictions), len(lin_predictions))
lstm_predictions = lstm_predictions[:min_length]
lin_predictions = lin_predictions[:min_length]


# In[26]:


lstm_future_predictions = []
last_sequence = X[-1].reshape(1, seq_length, 1)
for _ in range(10):
    lstm_pred = lstm_model.predict(last_sequence)[0, 0]
    lstm_future_predictions.append(lstm_pred)
    lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))


# In[28]:


recent_data = data['Close'].values[-3:]
lin_future_predictions = []
for _ in range(10):
    lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
    lin_future_predictions.append(lin_pred)
    recent_data = np.append(recent_data[1:], lin_pred)
lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))


# In[29]:


hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)


# In[30]:


future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM Predictions': lstm_future_predictions.flatten(),
    'Linear Regression Predictions': lin_future_predictions.flatten(),
    'Hybrid Model Predictions': hybrid_future_predictions.flatten()
})
print(predictions_df)


# In[35]:


import matplotlib.pyplot as plt

# Assuming your prediction data is defined
plt.figure(figsize=(12, 6))
plt.plot( lstm_predictions, label='LSTM Predictions', color='blue')
plt.plot( lin_predictions, label='Linear Regression Predictions', color='green')
plt.plot( hybrid_predictions, label='Hybrid Model Predictions', color='red')

plt.title('Predictions Comparison')
plt.xlabel('Time')
plt.ylabel('Predicted Values')
plt.legend()  # Add a legend
plt.grid(True)  # Add a grid for better readability
plt.show()  # Show the plot


# In[ ]:




