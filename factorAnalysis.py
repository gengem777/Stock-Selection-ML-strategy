#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:21:48 2020

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
import scipy.stats as stats

import statsmodels.api as sm

import pyfolio as pf
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


from collections import Counter
import copy

class FactorAnalysis():
    def __init__(self):
        self.freq = '1W' #set frequency
        
        #set ptice and index
        self.stock_prices = pd.read_csv('data2/HS300plus.csv', index_col=0)
        self.stock_codes = self.stock_prices.columns
        self.stock_index = pd.read_csv('data/HS300_index.csv', index_col=0)
        self.stock_prices.index = pd.to_datetime(self.stock_prices.index)
        self.stock_index.index = pd.to_datetime(self.stock_index.index)
        self.stock_prices = self.stock_prices.ffill(axis=0).fillna(0)
        self.stock_index = self.stock_index.ffill(axis=0).fillna(0)
        self.stock_returns = self.stock_prices.pct_change().fillna(0)
        self.index_returns = self.stock_index.pct_change().fillna(0)
        self.stocks_r_res = self.stock_prices.resample(self.freq).last().pct_change().fillna(0) #周频收益率
        self.index_r_res = self.stock_index.resample(self.freq).last().pct_change().fillna(0) #周频指数收益率
        
        #set factors
        self.f_name = ['momentum', 'gross_profit_ttm', 'market_cap']
        self.f_list = []
        self.factors = {}
        for name in self.f_name:  #put factors into a list and a dict with their name
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0).ffill(axis=0).fillna(0)
            self.f_list.append(f)
            self.factors.update({name:f})
        
        self.factors_res = {}  #dict for resampled factors
        for i in range(len(self.f_name)):
            f = copy.deepcopy(self.factors[self.f_name[i]])
            f.index = pd.to_datetime(f.index)
            f_ = f.resample(self.freq).last()#.fillna(axis=0,method='ffill')
            self.factors_res.update({self.f_name[i]:f_})
        
        #process z-score winsorize
        for name in self.f_name:
            self.factors_res[name] = self.factors_res[name].apply(lambda x: self.MAD(x,3), axis = 0)
            self.factors_res[name] = self.factors_res[name].apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis = 0)
            self.factors_res[name] = self.factors_res[name].fillna(0)
        
        
        self.stocks_r_res = self.stocks_r_res.reindex(columns=self.factors_res['market_cap'].columns)
        #self.factors_res['market_cap'] = self.factors_res['market_cap'].reindex(columns=self.stocks_r_res.columns)
        
        #sectors
        self.sectors = ['HY007', 'HY005', 'HY008', 'HY004', 'HY003', 'HY006',\
               'HY002', 'HY011', 'HY010', 'HY001', 'HY009']

        self.sector_df = {}
        for name in self.sectors:
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0)
            self.sector_df.update({name:f})
            

    def MAD(self, series,n):
        median = series.quantile(0.5)
        diff_median = ((series - median).abs()).quantile(0.5)
        max_range = median + n * diff_median
        min_range = median - n * diff_median
        return np.clip(series, min_range, max_range)
    
    def three_sigma(self, series,n):
        mean = series.mean()
        std = series.std()
        max_range = mean + n * std
        min_range = mean - n * std
        return np.clip(series, min_range, max_range)
    
      
    '''   
    HS300 = pd.read_csv('data2/HS300plus.csv', index_col=0)
    stocks = HS300.columns #剔除了ST股票的，每个时期都被剔除
    HS300_index = pd.read_csv('data/HS300_index.csv', index_col=0)
    HS300.index = pd.to_datetime(HS300.index)
    HS300_index.index = pd.to_datetime(HS300_index.index)
    freq = '1W'
    '''
    
    def FactorReturn_v1(self, factor_name):
        '''
        f_list = []
        factors = {} 
        for name in f_name:
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0)
            f_list.append(f)
            factors.update({name:f})
        
        
        for name in f_name:
            factors[name] = factors[name].apply(lambda x: mad(x,3), axis = 0)
            factors[name] = factors[name].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
            factors[name] = factors[name].fillna(0)
        
        factors_ = {} #what we need
        for i in range(len(f_name)):
            f = copy.deepcopy(factors[f_name[i]])
            f.index = pd.to_datetime(f.index)
            f_ = f.resample(freq).last().fillna(0)
            factors_.update({f_name[i]:f_})
        
        HS300_ = HS300.resample(freq).last() #what we need
        #HS300index_ = HS300_index.resample(freq).last() #what we need
        '''
        assert factor_name[0] in self.f_name
            
        
        factor_return = pd.DataFrame(0, index = self.factors_res[factor_name[0]].index, columns = [factor_name[0]])
        
        T = factor_return.shape[0]
        #COLUMNS = list(set(stocks).intersection(set(factors[f_name[0]].columns.tolist())))
        #COLUMNS1 = list(set(COLUMNS).intersection(set(factors_[f_name[1]].columns.tolist())))
        #HS300_plus = HS300_[COLUMNS1]
        #HSreturns = HS300_plus.pct_change().fillna(0)
        HSreturns_pre_target = self.stocks_r_res.shift(1).fillna(0)
        factor_plus = self.factors_res[factor_name[0]]
        file_name = 'factors/market_cap_extend'+'.csv'
        market_cap = pd.read_csv(file_name, index_col=0).ffill(axis=0).fillna(0)
        print(market_cap)
        print(HSreturns_pre_target)
        
        epls = 1e-7
        for t in range(2, T):
            y = HSreturns_pre_target.iloc[t,:].values
            X = factor_plus.iloc[t,:].values
            X = sm.add_constant(X)
            sqrt_cap = np.sqrt(market_cap.iloc[t,:].values)
            w = sqrt_cap/(np.sum(sqrt_cap)+epls)
            mod_wls = sm.WLS(y, X, weights=w)
            res_wls = mod_wls.fit()
            factor_return.iloc[t,:] = res_wls.params[1]
            #print(y, X, w)
        #factor_return['date'] = factor_return.index
        #factor_return.set_index('date', inplace=True)
        #pf.create_returns_tear_sheet(factor_return[f_name[0]
        print(factor_return[factor_name[0]])
        plt.figure(figsize = (10, 5))
        compare = (1+factor_return[factor_name[0]]).cumprod().ffill(axis = 0)
        ax1 = compare.plot(figsize = (10,5))
        ax1.legend(loc='upper left')
        fig = ax1.get_figure()
        fig_name = 'FacReturnV1/'+factor_name[0] + '.eps'
        fig.savefig(fig_name)
        
        
        #return factor_return
    
    #FactorReturn_v1(f_name)
    
    def FactorReturn_v2(self, factor_name, start_date, end_date, neu = False): 
        '''
        date format: YYYY-MM-DD
        
        f_list = []
        factors = {} 
        for name in f_name:
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0)
            f = f.loc[start_date:end_date,:]
            f_list.append(f)
            factors.update({name:f})
        
        
        for name in f_name:
            factors[name] = factors[name].apply(lambda x: mad(x,3), axis = 0)
            factors[name] = factors[name].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
            factors[name] = factors[name].fillna(axis=0,method='ffill')
        if neu:
            f_neu = Neutralize(factors[name])
            factors[name] = f_neu
            
        factors_ = {} #what we need
        for i in range(len(f_name)):
            f = copy.deepcopy(factors[f_name[i]])
            f.index = pd.to_datetime(f.index)
            f_ = f.resample(freq).last().fillna(axis=0,method='ffill')
            factors_.update({f_name[i]:f_})
        
        HS300_ = HS300.loc[start_date:end_date,:].resample(freq).last().fillna(axis=1,method='ffill')
        HSreturns = HS300_.pct_change()
        
        '''
        t1 = start_date
        t2 = end_date
        r = self.stocks_r_res.loc[t1:t2,:]
        factors_res = self.factors_res[factor_name[0]].loc[t1:t2,:]
        #mv = self.factors_res['market_cap'].loc[t1:t2,:]
        if neu:
            factors_res = self.Neutralize(factors_res)
        
        layers = ['1','2','3','4','5','6','7','8','9','10','L-S']
        factor_return = pd.DataFrame(0, index = factors_res.index, columns = layers)
        #COLUMNS = list(set(stocks).intersection(set(factors[f_name[0]].columns.tolist())))
        T = factor_return.shape[0]
        N = len(factors_res.columns.tolist())
        #transaction_rate = 0.003
        ratio = 0.1
        for layer in layers:
            df_weights = pd.DataFrame(0, index=factors_res.index, columns=self.factors_res[factor_name[0]].columns.tolist())
            for t in range(T-1):
                if t % 4 == 0:
                    fac = factors_res.iloc[t,:]
                    x = (-fac).argsort(fac)
                    fac_ = fac[x]
                    if layer == 'L-S':
                        fac_top_name = fac_[:int(N*ratio)].index.tolist()
                        fac_bottom_name = fac_[int(N*(1-ratio)):].index.tolist()
                        weight1 = np.ones(len(fac_top_name))/len(fac_top_name)
                        weight2 = -np.ones(len(fac_bottom_name))/len(fac_bottom_name)
                        df_weights.loc[df_weights.index[t],fac_top_name] = weight1
                        df_weights.loc[df_weights.index[t],fac_bottom_name] = weight2
                    elif layer == '10':
                        fac_name =  fac_[int(N*(1-ratio)):].index.tolist()
                        weight = np.ones(len(fac_name))/len(fac_name)
                        df_weights.loc[df_weights.index[t],fac_name] = weight 
                    else:
                        i = layers.index(layer)
                        fac_name =  fac_[int(i*N*ratio):int((i+1)*N*ratio)].index.tolist()
                        weight = np.ones(len(fac_name))/len(fac_name)
                        df_weights.loc[df_weights.index[t],fac_name] = weight
                                    
                    '''
                    if t == 0:
                        Tcost = np.sum(abs(df_weights.iloc[t,:])*transaction_rate)
                    else:
                        weight_minus = df_weights.iloc[t-1,:]*(1+HSreturns[COLUMNS].loc[df_weights.index[t], :])\
                        /(np.sum(df_weights.iloc[t-1,:]*(1+HSreturns[COLUMNS].loc[df_weights.index[t], :])))
                        Tcost = np.sum(abs(df_weights.iloc[t,:] - weight_minus)*transaction_rate)
                    '''
                else:
                    '''
                    if layer == 'L-S':
                   # df_weights.loc[df_weights.index[t],:] = df_weights.iloc[t-1,:]*(1+HSreturns[COLUMNS].loc[df_weights.index[t], :]/52)\
                   #     /(np.sum(df_weights.iloc[t-1,:]*(1+HSreturns[COLUMNS].loc[df_weights.index[t], :]/52)))
                        df_weights.loc[df_weights.index[t],fac_top_name] = df_weights.iloc[t-1,:][fac_top_name]*(1 + HSreturns[fac_top_name].loc[df_weights.index[t], :])/\
                         np.sum(df_weights.iloc[t-1,:][fac_top_name]*(1 + HSreturns[fac_top_name].loc[df_weights.index[t], :]))
                        df_weights.loc[df_weights.index[t],fac_bottom_name] = df_weights.iloc[t-1,:][fac_bottom_name]*(1 + HSreturns[fac_bottom_name].loc[df_weights.index[t], :])/\
                         np.sum(df_weights.iloc[t-1,:][fac_bottom_name]*(1 + HSreturns[fac_bottom_name].loc[df_weights.index[t], :]))
                    else:
                        df_weights.loc[df_weights.index[t],fac_name] = df_weights.iloc[t-1,:][fac_name]*(1 + HSreturns[fac_name].loc[df_weights.index[t], :])/\
                         np.sum(df_weights.iloc[t-1,:][fac_name]*(1 + HSreturns[fac_name].loc[df_weights.index[t], :]))
                    '''
                    df_weights.loc[df_weights.index[t],:] = df_weights.iloc[t-1,:]
                #return_before_cost = np.sum(df_weights.iloc[t,:] * HSreturns[COLUMNS].iloc[t+1,:])
                #return_after_cost = return_before_cost #- Tcost
                #factor_return.iloc[t,0] = return_after_cost
                factor_return.loc[factor_return.index[t],layer] = np.sum(df_weights.iloc[t,:] * r.iloc[t+1,:])
        #factor_return['date'] = factor_return.index
        #factor_return.set_index('date', inplace=True)
        #pf.create_returns_tear_sheet(factor_return[f_name[0]])
        plt.figure(figsize = (10, 5))
        compare = (1+factor_return).cumprod()
        ax1 = compare.plot(figsize = (10,5))
        ax1.legend(loc='upper left')
        fig = ax1.get_figure()
        fig_name = 'FacReturnV2/'+factor_name[0] + '.eps'
        fig.savefig(fig_name)
        #return factor_return
        
        
    
    def _IC(self, return_list, factor_list):
        return np.corrcoef(return_list, factor_list)
    
    def _Rank_IC(self, return_list, factor_list):
        ordered_r = np.argsort(return_list)
        ordered_f = np.argsort(factor_list)
        return np.corrcoef(ordered_r, ordered_f)
    
    def FactorICtest(self, factor_name):
        '''
        f_list = []
        factors = {} 
        for name in f_name:
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0)
            f_list.append(f)
            factors.update({name:f})
        
        
        for name in f_name:
            #if factors[name].shape[0] > 2621:
             #   factors[name] = factors[name].drop(index = factors[name].index[2621:])
            factors[name] = factors[name].apply(lambda x: mad(x,3), axis = 0)
            factors[name] = factors[name].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
            factors[name] = factors[name].fillna(0)
        
        factors_ = {} #what we need
        for i in range(len(f_name)):
            f = copy.deepcopy(factors[f_name[i]])
            f.index = pd.to_datetime(f.index)
            f_ = f.resample(freq).last().fillna(0)
            factors_.update({f_name[i]:f_})
        
        HS300_ = HS300.resample(freq).last() 
        HSreturns = HS300_.pct_change().fillna(0)
        '''
        
        IC_s = pd.DataFrame(0,self.stocks_r_res.index,columns=['IC', 'Rank_IC'])
        f = self.factors_res[factor_name[0]]
        
        for t in range(1, self.stocks_r_res.shape[0]-1):
            r = self.stocks_r_res.iloc[t+1,:].values
            f_ = f.iloc[t,:].values
            
            IC_s.iloc[t,0] = self._IC(r, f_)[0,1]
            IC_s.iloc[t,1] = self._Rank_IC(r, f_)[0,1]
        
        plt.figure(figsize = (10, 5))
        plt.bar(IC_s.index,IC_s['Rank_IC'],width = 1, facecolor = 'lightskyblue')
        plt.plot(IC_s['Rank_IC'].rolling(window=52).mean(), color = 'orange')
        plt.title('Rank IC')
        plt.legend([factor_name[0]])
        fig_name = 'IC_figures/' + factor_name[0] + '.eps'
        plt.savefig(fig_name)
        plt.show()
    
    
    
    
    def Neutralize(self, factor):
    
        market_cap = ['market_cap']
        
        t1 = factor.index[0]
        t2 = factor.index[-1]
        
        s_m = self.sectors + market_cap
        sm_list = []
        sm_fac = {} 
        for name in s_m:
            file_name = 'factors/'+name+'_extend'+'.csv'
            f = pd.read_csv(file_name, index_col=0)
            f = f.loc[t1:t2,:]
            sm_list.append(f)
            sm_fac.update({name:f})
        for name in s_m[:-1]:
            sm_fac[name] = sm_fac[name].fillna(axis=0,method='ffill')
            sm_fac[name] = sm_fac[name].fillna(0)
        #sm_fac['market_cap'] = sm_fac['market_cap'].apply(lambda x: mad(x,3), axis = 0)
        #sm_fac['market_cap'] = sm_fac['market_cap'].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis = 0)
        #sm_fac['market_cap'] = sm_fac['market_cap'].apply(lambda x:np.log(x))
        sm_fac['market_cap'] = sm_fac['market_cap'].ffill(axis=0).fillna(1)
        
        
        neu_factor = pd.DataFrame(0, index=factor.index, columns=factor.columns.tolist())
        
        T = neu_factor.shape[0]
        
        formula = 'exposure~HY007+HY005+HY008+HY004+HY003+HY006+HY002+HY011+HY010+HY001+HY009+np.log(market_cap)'
        for t in range(T):
            I = sm_list[0].iloc[t,:].index
            X_ = pd.DataFrame(sm_list[0].iloc[t,:].values, \
                              index = I.tolist(), columns = [s_m[0]])
            for i in range(1, len(sm_list)):
                X_ = pd.concat([X_,sm_list[i].iloc[t,:]], axis = 1)
            X_.columns = s_m
            X_['exposure'] = factor.iloc[t,:]
            Data = X_.iloc[X_.index.str[-2:] != '.1'].fillna(1)
            #print(Data)
            import statsmodels.formula.api as ssm
            result = ssm.ols(formula = formula, data = Data).fit()
            neu_factor.iloc[t,:] = result.resid
        print(neu_factor)
        return neu_factor

if __name__ == '__main__':
    '''
    # IC 测试
    all_factors = pd.read_csv('all_factors.csv', index_col=0)
    all_factors = all_factors.drop([0],axis = 0)
    factors_name = all_factors['factor'].tolist()
    factors_name_1 = factors_name[170:] #for debug
    for name in factors_name_1:
        f_name = [name, 'market_cap']
        FactorICtest(f_name)
        
    '''
    start_date = '2017-01-01'
    end_date = '2020-10-25'
    FA = FactorAnalysis()
    FA.FactorReturn_v1(['momentum'])
    #FA.FactorReturn_v2(['momentum'], start_date, end_date, neu=False)
    '''
    all_factors = pd.read_csv('all_factors.csv', index_col=0)
    all_factors = all_factors.drop([0],axis = 0)
    factors_name = all_factors['factor'].tolist()
    for name in factors_name: 
        FactorReturn_v2([name])
    '''















