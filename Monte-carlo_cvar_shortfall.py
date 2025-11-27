#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:53:23 2025

@author: rbarcrosspbar
"""


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats as scpy
from scipy.stats import shapiro
from scipy.stats import norm

plt.rcParams["figure.figsize"] = (15, 8)

days = input("Enter the number of days ")

ed = dt.date.today()

sd = ed - pd.Timedelta(days=5000)

df = yf.download("NVDA", start=sd, end=ed, interval="1d")

df["Returns"] = (df['Close']/df['Close'].shift(1))-1
df["Log_Rets"] = np.log(1 + df["Returns"])

df.dropna(inplace=True)
plt.hist(df["Returns"], bins=100)
plt.xlim(left=-0.15, right=0.15)
plt.axvline(0.00, color="red")
plt.title(f"Histogram of Daily returns for {days} days" )


mean_ret = np.mean(df["Returns"])
std_ret = np.std(df["Returns"])

ann_std = np.sqrt(252)*std_ret
ann_var = ann_std**2

print("Mean Return, Standard Deviation of Return, Annualized Standard Deviation, Annualized Variancein%:",round(mean_ret*100,3),round(std_ret*100,3),round(ann_std*100,3),round(ann_var*100,3))


skwnss = scpy.skew(df["Returns"])
krtsis = scpy.kurtosis(df["Returns"])

print(round(skwnss,2),round(krtsis,2))

statistic, p_value = shapiro(df["Returns"])

if p_value <= 0.05:
    print("Null hypothesis of normality is rejected, the dataset is not normally distributed.")
else:
    print("Null hypothesis of normality is accepted, the dataset is normally distributed.")


var_level = 95


sorted_rets = np.sort(np.array(df["Returns"]))

pos = int(len(sorted_rets)*(1-var_level/100))

print(round(sorted_rets[pos],4))

print(round(np.percentile(df["Returns"], 5),4))


plt.hist(sorted_rets, bins=100)
plt.xlim(left=-0.15, right=0.15)
plt.axvspan(-0.15,round(sorted_rets[pos],4), color="red", alpha=0.2)
plt.title(f"Histogram of Daily returns for {days} days" )

cvar = np.mean(sorted_rets[sorted_rets<=round(sorted_rets[pos],4)])

print(cvar)

alpha = 1-var_level*0.01

var_95 = round(scpy.norm.ppf(alpha, mean_ret, std_ret),3)


cvar_95 = round(mean_ret + (scpy.norm.pdf(scpy.norm.ppf(alpha)) / alpha) * std_ret,3)


print("Parametric VaR(95):"+str(var_95))
print("Parametric CVaR(95):"+str(cvar_95))


def plot_var(array):
  d = pd.DataFrame(abs(array))
  d[1].plot(xlabel='Time', ylabel='Forecasted VaR-95', title = "Time scaled VaR")
  plt.show()

VaR_arr = np.empty([252, 2])

for t in range(1,253):
  VaR_arr[t-1,0] = t
  VaR_arr[t-1,1] = var_95 * np.sqrt(t)

plot_var(VaR_arr)



num_simulations = 1000  
num_days = 5000  
last_price = 100

z = np.random.normal(size=(num_days,num_simulations))

drift = np.mean(df["Log_Rets"]) - 0.5*np.var(df["Log_Rets"])

r = drift + (std_ret * z)

daily_returns = np.exp(r)

price_paths = np.zeros_like(daily_returns)
price_paths[0] = last_price


for t in range(1, num_days):
    price_paths[t] = price_paths[t-1] * daily_returns[t]


price_paths_df = pd.DataFrame(price_paths)

plt.figure(figsize=(10, 6))
plt.plot(price_paths_df)
plt.title('Monte Carlo Simulation of Asset Price Paths')
plt.xlabel('Days')
plt.ylabel('Simulated Price')
plt.show()

final_prices = price_paths_df.iloc[-1]

confidence_level = 0.95

var_price = np.percentile(final_prices, (1 - confidence_level) * 100)
print(f"VaR (95%) Price: ${var_price:.2f}")

cvar_prices = final_prices[final_prices <= var_price]

cvar_price = cvar_prices.mean()

print("CVar is "+str(cvar_price))