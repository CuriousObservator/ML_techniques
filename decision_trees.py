#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 12:47:50 2025

@author: rbarcrosspbar
"""

import numpy as np
import pandas as pd 
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns
from utility import get_target_features
from utility import get_metrics
from sklearn import tree
import graphviz

file = "/home/rbarcrosspbar/EPAT/Mini-Projects/JPM_data/JPM_2017_2019.csv"

data = pd.read_csv(file, index_col=0)

data.index = pd.to_datetime(data.index)

data.close.plot()

plt.xlabel("Date-->")
plt.ylabel("Price-->")

plt.show()

#%%

y, x = get_target_features(data)

split = int(0.8*len(x))

x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5)

model = model.fit(x_train, y_train)

prediction = model.predict(x_test)

get_metrics(y_test, prediction)

#%%

graph_data = tree.export_graphviz(model, out_file=None, filled=True, feature_names=x_train.columns)

graphviz.Source(graph_data)
