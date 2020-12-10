#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:52:39 2020

@author: zhangwenyong
"""

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import os
import datetime
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from collections import Counter
import copy


def mad(series,n):
    median = series.quantile(0.5)
    diff_median = ((series - median).abs()).quantile(0.5)
    max_range = median + n * diff_median
    min_range = median - n * diff_median
    return np.clip(series, min_range, max_range)

def three_sigma(series,n):
    mean = series.mean()
    std = series.std()
    max_range = mean + n * std
    min_range = mean - n * std
    return np.clip(series, min_range, max_range)

def MLtrain(y_data, X_data, method = 'ensemble', gridsearch = False):
    y_ = y_data.values
    if X_data.values.ndim == 1:
        X_ = X_data.values.reshape(-1,1)
    else:
        X_ = X_data.values
    clf_list = [#('svm', svm.SVC(probability=True)),
            #('logistic regression', LogisticRegression()),
            ('knn', KNeighborsClassifier()),
            ('naive bayes', GaussianNB()),
            ('Adaboost', AdaBoostClassifier()),
            ('GradientBoosting', GradientBoostingClassifier())]
    #set model
    if method == 'naive bayes':
        model = GaussianNB()
    elif method == 'knn':
        model = KNeighborsClassifier()
    elif method == 'svm':
        model = svm.SVC(probability=True)
    elif method == 'Adaboost':
        model = AdaBoostClassifier()
    elif method == 'RandomForest':
        model = RandomForestClassifier()
    elif method == 'GradientBoosting':
        model = GradientBoostingClassifier()
    elif method == 'LogisticRegression':
        method = LogisticRegression()
    elif method == 'ensemble':
        model = VotingClassifier(clf_list, voting='soft')

    model.fit(X_, y_) 
    return model

def min_variance(r, lower_w = 1e-5, upper_w = 0.3):
    avg_returns = np.matrix(r.mean())
    n = avg_returns.shape[1]
    #print(n)
    w = np.matrix( np.ones(n)/n)
    mu = w.dot(avg_returns.T)+0.005
    S = np.cov(r, rowvar = False)
    #print(S.shape)
    var = lambda w: w.dot(S.dot(w.T))
    w_bound = [(lower_w, upper_w) for i in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},\
        {'type': 'ineq', 'fun': lambda w: np.array(np.dot(avg_returns,w.T) - mu)[0,0]},\
       )
    optimal_w = minimize(var, w, method = 'SLSQP', constraints = cons, bounds = w_bound)
    
    return optimal_w.x

def Get_ICs(factor_name):
    IC_s = pd.DataFrame(0,index=HSreturns.index,columns=['IC', 'Rank_IC'])
    def _IC(return_list, factor_list):
        return np.corrcoef(return_list, factor_list)

    def _Rank_IC(return_list, factor_list):
        ordered_r = np.argsort(return_list)
        ordered_f = np.argsort(factor_list)
        return np.corrcoef(ordered_r, ordered_f)

    f = factors_weekly[factor_name].shift(1).fillna(0)

    for t in range(1, T):
        r = HSreturns.iloc[t,:].values
        f_ = f.iloc[t,:].values
        IC_s.iloc[t,0] = _IC(r, f_)[0,1]
        IC_s.iloc[t,1] = _Rank_IC(r, f_)[0,1]

    IC_s[-52:].plot(figsize = (16,8),kind='bar')

HS300 = pd.read_csv('data2/HS300plus.csv', index_col=0)
stocks = HS300.columns
HS300_index = pd.read_csv('data/HS300_index.csv', index_col=0)
HS300.index = pd.to_datetime(HS300.index)
HS300_index.index = pd.to_datetime(HS300_index.index)


#f_name = ['momentum','EMA5','EMAC10','MAC5','MAC10', 'book_to_price_ratio']
f_name = ['EMA5']
f_list = []

factors = {}
for name in f_name:
    file_name = 'data2/'+name+'_extend'+'.csv'
    f = pd.read_csv(file_name, index_col=0)
    f_list.append(f)
    factors.update({name:f})


for name in f_name:
    factors[name] = factors[name].apply(lambda x: three_sigma(x,3), axis = 0)
    factors[name] = factors[name].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
    factors[name] = factors[name].fillna(0)

factors_weekly = {}
for i in range(len(f_name)):
    f = copy.deepcopy(factors[f_name[i]])
    f.index = pd.to_datetime(f.index)
    f_weekly = f.resample('1W').last().fillna(0)
    factors_weekly.update({f_name[i]:f_weekly})

HS300_weekly = HS300.resample('1W').last()
HS300index_weekly = HS300_index.resample('1W').last()


HSreturns = HS300_weekly.pct_change().fillna(0)
HSIreturns = HS300index_weekly.pct_change().fillna(0) 

T, N = HSreturns.shape
rate = 0.3 #30%
select_ratio = 0.1
y_labels = pd.DataFrame(np.zeros((T, N)), index=HSreturns.index, columns=stocks)
#labeling
for t in range(T):
    x = HSreturns.iloc[t,]
    x1 = x.argsort(x)
    low_stocks = x[x1][:int(N*rate)].index.tolist()
    high_stocks = x[x1][int(N*(1-rate)):].index.tolist()
    y_labels.iloc[t][low_stocks] = -1
    y_labels.iloc[t][high_stocks] = 1
y_pred_labels = y_labels.shift(1).fillna(0)

#设定窗口长度
train_length = 104 #训练集：2年
test_length = 26 #测试集：半年
change_position_length = 4 #调仓频率：月度
mean_variance_length = 10 #均值方差窗口：10周
transaction_rate = 0.003

backtest = pd.DataFrame()
backtest = pd.DataFrame(0,index=HSreturns.index,columns=['RandomForest'])
#回测数据框初始化
a=np.load('a.npy', allow_pickle=True)
baskets = a.tolist()

MLmethods = ['RandomForest']#, 'Adaboost','naive bayes','LogisticRegression', 'eq'] 

weights = np.zeros(N)
weights_0 = np.zeros(N)
weights_plus = np.zeros(N)
weights_0_plus = np.zeros(N)

for method in MLmethods:
    backtest[method] = 0
backtest['eq_portfolio'] = 0

df_weight = pd.DataFrame(0,index=HSreturns.index,columns=HSreturns.columns.tolist())
df_weight_0 = pd.DataFrame(0,index=HSreturns.index,columns=HSreturns.columns.tolist())
print(backtest.head())
for i in range(len(MLmethods)):
    method = MLmethods[i]
    model = None
    select_ = [] #被选中的股票代码
    counter_of_list = 0
    basket = []
    for t in range(train_length, T-1):
        if (t - train_length) % test_length == 0: #训练
            if counter_of_list < len(basket):
                basket = baskets[counter_of_list]
            else:
                basket = baskets[len(baskets) - 1]
            basket_ = list(set(basket).intersection(set(factors[f_name[0]].columns.tolist())))
            y_with_length = y_pred_labels[basket_].iloc[(t - train_length):t,:]
            X_data = factors_weekly[f_name[0]][basket_].iloc[(t - train_length):t,:].stack()
            #X_data.columns = f_name
            for i in range(1,len(f_name)):
                X_data = pd.concat([X_data,factors_weekly[f_name[i]][basket_].iloc[(t - train_length):t,:].stack()], axis = 1)
            y_data = y_with_length.stack()
            X_data.columns = f_name
            model = MLtrain(y_data, X_data,method = method)
            print(method,'train time ',t)
            counter_of_list += 1
                
        if (t - train_length) % change_position_length == 0: #选股
            I = factors_weekly[f_name[0]][basket_].iloc[t,:].index
            test_data_thisWeek= pd.DataFrame(factors_weekly[f_name[0]][basket_].iloc[t,:].values, \
                                              index = I.tolist(), columns = [f_name[0]])
            for i in range(1,len(f_name)):
               test_data_thisWeek = pd.concat([test_data_thisWeek,factors_weekly[f_name[i]][basket_].iloc[t,:]], axis = 1)
            test_data_thisWeek.columns = f_name 
            
            if test_data_thisWeek.values.ndim == 1:
                test_data_features = test_data_thisWeek.values.reshape(-1,1)
            else:
                test_data_features = test_data_thisWeek.values
            test_data_thisWeek['predict_label'] = model.predict(test_data_features)
            test_data_thisWeek['predict_proba'] = model.predict_proba(test_data_features)[:,2]
            prob = test_data_thisWeek['predict_proba'] 
            x1 = (-prob).argsort(prob) #排序，选前10%
            select = prob[x1][:int(select_ratio*N)].index.to_list()
            select_ = list(set(select).intersection(set(basket_)))#由于部分股票的名称跟因子对应的名称不一致，因此取交集。
            data_r = HSreturns[select_].iloc[t-mean_variance_length:t,:]
            #print(data_r.shape)
            '''
            if i != len(MLmethods)-1:
             weights = min_variance(data_r) #调仓
            else:
            '''
            weights = np.ones(len(select_))/len(select_)
            
            #print('weights: ', weights)
            #print('change position',t, ' stock seleted ', len(select_))
            #backtest.iloc[t,i] = np.sum(weights * HSreturns[select_].iloc[t,:].values)
            #print(backtest.iloc[t, i])
            #backtest.iloc[t, 2] = np.mean(HSreturns.iloc[t,:].values)
       # else:
        
            #backtest.iloc[t,i] = np.sum(weights * HSreturns[select_].iloc[t,:].values)
            #backtest.iloc[t, 2] = np.mean(HSreturns.iloc[t,:].values)
        
        #return calculation and rebalance portfolio
        df_weight.loc[HSreturns.index[t], select_] = weights
        return_before_Tcost = np.sum(weights * HSreturns[select_].iloc[t+1,:].values)
        weights_plus = df_weight.loc[HSreturns.index[t], :].values
        if (t - train_length) % change_position_length == 0:
            Tcost = np.sum(abs(weights_plus - weights_0_plus)*transaction_rate)
            return_after_Tcost = return_before_Tcost - Tcost
        else:
            return_after_Tcost = return_before_Tcost
        weights_0 = weights*(1+HSreturns.loc[HSreturns.index[t], select_])/(np.sum(weights*(1+HSreturns.loc[HSreturns.index[t], select_])))
        df_weight_0.loc[HSreturns.index[t], select_] = weights_0
        weight_0_plus = df_weight_0.loc[HSreturns.index[t], :].values
        backtest.iloc[t,i] = return_after_Tcost
    backtest[method] = backtest.iloc[:,i]
    print(backtest[method])
#for t in range(train_length, T-1):
#    backtest.iloc[t, 5] = np.mean(HSreturns.iloc[t,:].values)






plt.figure()
#((backtest['ensemble'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='r')
#((backtest['Adaboost'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='y')
((backtest['RandomForest'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='c')
#((backtest['svm'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='m')
#((backtest['naive bayes'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='indigo')
#((backtest['logistic regression'][train_length:]+1).cumprod()-1).plot(figsize=(15,7),color='salmon')
#((backtest['knn'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='salmon')
#((backtest['eq'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='g')
((HSIreturns['000300.XSHG'][train_length:]+1).cumprod()).plot(figsize=(15,7),color='black')
plt.ylabel('Cumulative Returns')
plt.legend(MLmethods + ['HS300 Index'])
plt.show()


import statsmodels.api as sm
from statsmodels import regression
from prettytable import PrettyTable

def MaxDrawdown(return_list):
    #a = np.maximum.accumulate(return_list)
    l = np.argmax((np.maximum.accumulate(return_list) - return_list) /np.maximum.accumulate(return_list))
    k = np.argmax(return_list[:l])
    return (return_list[k] - return_list[l])/(return_list[l])

def Performance(method = 'ensemble'):
    table = PrettyTable(['performance', 'value'])

    #max drawdown
    mdd = MaxDrawdown((backtest[method][train_length:]+1).cumprod())
    table.add_row(['MaxDrawdown', mdd.astype(np.float16)])
    #time horizon
    t = (backtest[method].index[T-1]-backtest[method].index[train_length]).days
    #annual return
    anul_return = (backtest[method][train_length:]+1).cumprod()[-1]**(1/t*255)-1
    #annual vol
    anul_vol = backtest[method].std()*(52**(1/2))
    #Sharpe Ratio
    spr = anul_return/anul_vol
    table.add_row(['Sharpe Ratio', spr.astype(np.float16)])
    #Sortino Ratio
    down_return = backtest[method][backtest[method]<0]
    anul_downvol = down_return.std()*(52**(1/2))
    sorr = anul_return/anul_downvol
    table.add_row(['Sortino Ratio', sorr.astype(np.float16)])
    #beta & alpha
    y = backtest[method].iloc[train_length:-1]
    x = HSIreturns['000300.XSHG'].iloc[train_length:-1]
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y,x).fit()
    beta = model.params[1]
    alpha = model.params[0]
    table.add_row(['beta', beta.astype(np.float16)])
    table.add_row(['alpha', alpha.astype(np.float16)])
    #IR
    cum_return = ((backtest[method][train_length:]+1).cumprod() - \
    (HSIreturns['000300.XSHG'][train_length:]+1).cumprod())[-1]**(1/t*255)
    alpha_ = backtest[method].iloc[train_length:] - HSIreturns['000300.XSHG'].iloc[train_length:]
    alpha_vol = alpha_.std()*(52**(1/2))
    ir = cum_return/alpha_vol
    table.add_row(['information rate', ir.astype(np.float16)])
    print(table.get_string(title=method))

for method in MLmethods:
    Performance(method)














































