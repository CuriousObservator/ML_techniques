#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 15:04:20 2025

@author: rbarcrosspbar
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import yfinance as yf
import requests
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# session = requests.Session()
# msf_tick = yf.Ticker("MSFT")
# df = msf_tick.history(start='2015-01-01', end='2024-01-01') #use this if yf doesn't give any trouble downloading the data
df = pd.read_csv("/home/rbarcrosspbar/Downloads/MSFT_1D.csv")
df.set_index('datetime', inplace=True)
df.sort_values(by="datetime", ascending=True, inplace=True)
# df = yf.download("MSFT", start='2015-01-01', end='2024-01-01', session=session)
#%%

print(df.head())
#%%
# df.columns = df.columns.droplevel(1)
# print(df.head())
#%%
df["close"].plot()
#%%
prices_df = df["close"]
# print(prices_df.head())
prices = np.array(prices_df)
print(prices)
#%%
training_length = int(0.8*len(prices))
#%%
reshaped_prices = prices.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(reshaped_prices)
print(scaled_prices)
#%%
training_set = [i[0] for i in scaled_prices[:training_length]]
#%%    
training_set = np.array(training_set)
# print(training_set)

#%%
x_train = []
y_train = []

x_test = []
y_test = []


N = 60

for i in range(N, training_length):
    x_train.append(scaled_prices[i-N:i])
    y_train.append(scaled_prices[i])

for i in range(training_length+N, len(scaled_prices)):
    x_test.append(scaled_prices[i-N:i])
    y_test.append(scaled_prices[i])
    
x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
#%%    
import torch
import torch.nn as nn
#%%
n_timestamps = 60
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

class LSTM_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True)
    
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size).to(input_seq.device)
        lstm_out, (hn, cn) = self.lstm(input_seq, (h0, c0))
        final_timestep_output = lstm_out[:, -1, :]
        predictions = self.linear(final_timestep_output)
        
        return predictions

model = LSTM_Predictor(input_size, hidden_size, num_layers, output_size)

#%%

from torch.utils.data import TensorDataset, DataLoader

x_train_np = x_train.astype(np.float32)
y_train_np = y_train.astype(np.float32)

train_x_tensor = torch.from_numpy(x_train_np)
train_y_tensor = torch.from_numpy(y_train_np)

train_data = TensorDataset(train_x_tensor, train_y_tensor)

batch_size = 32

train_loader = DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle = True
    )

#%%

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 25

for epoch in range(epochs):
    for batch_sequences, batch_labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_sequences)
        loss = loss_function(y_pred, batch_labels)
        loss.backward()
        optimizer.step()

#%%
from torch.utils.data import TensorDataset, DataLoader

test_X_tensor = torch.from_numpy(x_test).float()
test_Y_tensor = torch.from_numpy(y_test).float()

#%%
test_data = TensorDataset(test_X_tensor, test_Y_tensor)

test_batch_size = 32 
test_loader = DataLoader(
    test_data, 
    batch_size=test_batch_size, 
    shuffle=False 
)
#%%
model.eval()

all_predictions = []
all_true_values = []

with torch.no_grad():
    for batch_sequences, batch_labels in test_loader:
        y_pred = model(batch_sequences)
        
        all_predictions.append(y_pred.cpu().numpy())
        all_true_values.append(batch_labels.cpu().numpy())


predictions_np = np.concatenate(all_predictions)
true_values_np = np.concatenate(all_true_values)
#%%
predictions_original = scaler.inverse_transform(predictions_np.reshape(-1, 1))
true_values_original = scaler.inverse_transform(true_values_np.reshape(-1, 1))

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(true_values_original, predictions_original))
mae = mean_absolute_error(true_values_original, predictions_original)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

#%%
from matplotlib.pyplot import figure
figure(figsize=(12,7))
plt.text(100, 550, "GOOD ESTIMATE", fontsize=14)
plt.text(700, 550, "OVER-ESTIMATED", fontsize=14)
plt.text(1280, 150, "GROSSLY-OVERESTIMATED", fontsize=12)
plt.plot(predictions_original, color="blue")
plt.plot(true_values_original, color="red")
plt.title('MSFT Price prediction using LSTM')
plt.xlabel('time-->')
plt.ylabel('Price-->')
plt.axvspan(0, 550, color='green', alpha=0.5)
plt.axvspan(550, 1250, color='yellow', alpha=0.5)
plt.axvspan(1250, 1750, color='red', alpha=0.5)

