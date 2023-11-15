#PLEASE DO NOT RUN THIS CODE
#This code is non-compiling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
 
# Assuming 'df' is your DataFrame with columns including 'timestamp', 'feature1', 'feature2', ..., 'value'
# Make sure the 'timestamp' column is in datetime format
 
scripts = ['script1','script2','script3','script4','script5','script6','script7','script8','script9','script10']
 
# Sort DataFrame by timestamp
df = pd.read_csv('/home/rock/hackathon/tem_dataset_collated_final.csv')
df['submit_time'] = pd.to_datetime(df['submit_time']).apply(lambda x: x.timestamp())
df['rel_date'] = pd.to_datetime(df['rel_date']).apply(lambda x: x.timestamp())
df['reserved_time'] = pd.to_datetime(df['reserved_time']).apply(lambda x: x.timestamp())
df['completed_time'] = pd.to_datetime(df['completed_time']).apply(lambda x: x.timestamp())
 
label_encoder = LabelEncoder()
df['tests_run'] = label_encoder.fit_transform(df['tests_run'])
 
print(df)
#df = df.sort_values('timestamp')
# Extract values from features and target variable
features = df.drop(['completed_time'], axis=1).values
target = df['completed_time'].values.reshape(-1, 1)
# Normalize the data
scaler_features = MinMaxScaler(feature_range=(0, 8))
scaler_target = MinMaxScaler(feature_range=(0, 1))
 
#scaler_features = normalize(features, axis=0, order=2)  # axis=0 normalizes along columns
#scaler_target = normalize(target, axis=0, order=2)
 
scaled_features = scaler_features.fit_transform(features)
scaled_target = scaler_target.fit_transform(target)
# Function to create sequences for time series data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        label = data[i+sequence_length:i+sequence_length+1]
        sequences.append((seq, label))
    return np.array(sequences)
# Hyperparameters
sequence_length = 10
epochs = 50
batch_size = 32
# Create sequences
sequences = create_sequences(np.hstack((scaled_features, scaled_target)), sequence_length)
# Split the data into train and test sets
split = int(0.8 * len(sequences))
train_sequences, test_sequences = sequences[:split], sequences[split:]
# Convert sequences to numpy arrays
X_train, y_train = np.array([seq[0] for seq in train_sequences]), np.array([seq[1] for seq in train_sequences])
X_test, y_test = np.array([seq[0] for seq in test_sequences]), np.array([seq[1] for seq in test_sequences])
# Reshape input for LSTM and GRU layers
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
# Build GRU model
gru_model = Sequential()
gru_model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2])))
gru_model.add(Dense(1))
gru_model.compile(optimizer='adam', loss='mean_squared_error')
# Train both models
lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
gru_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
# Evaluate both models on the test set
lstm_predictions = lstm_model.predict(X_test)
gru_predictions = gru_model.predict(X_test)
# Inverse transform predictions to original scale
lstm_predictions = scaler_target.inverse_transform(lstm_predictions)
gru_predictions = scaler_target.inverse_transform(gru_predictions)
# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'][split+sequence_length:], df['value'][split+sequence_length:], label='True Values')
plt.plot(df['timestamp'][split+sequence_length:], lstm_predictions, label='LSTM Predictions')
plt.plot(df['timestamp'][split+sequence_length:], gru_predictions, label='GRU Predictions')
plt.legend()
plt.show()