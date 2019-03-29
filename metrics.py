# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:39:13 2019

@author: andrey.babynin
"""

import pandas as pd
import numpy as np


'''
Metrics

highest values last n ticks
lowest values last n ticks
price crosess highest/lowest values (support resistance)
ema last n ticks
ema m crosses ema n
price relative to lowest / highest values
'''
'''
raw_data = pd.read_csv('GAZP_180101_181101_5min.csv', header = 0, sep =';')     
 
close_data = raw_data['<CLOSE>'] 
'''
class Metrics:
    
    def __init__(self, history):
        
        self.history = history # all history of prices
        self.levels = {'resistance': {}, 'support': {}, 'price return': {},
                       'ema': {}, 'volatility': {}
                       } # dictionary of eda statististics
        self.events = {'cross_resistance': {}, 'cross_support': {},
                       'cross ema from below': {},
                       'cross ema from above': {},
                       'price increase': 0
                        } # dictionary of statististics
    
    def eda(self, levels, axis = 0):
        for l in levels:
            self.levels['resistance'][l] = self.history[:-l].max()
            self.levels['support'][l] = self.history[:-l].min()
            self.levels['ema'][l] = np.ma.average(self.history[-l:], axis=0, 
                       weights=range(l))
            
            self.levels['price return'][l] = self.history[-1]/self.history[-l]-1
            self.levels['volatility'][l] = np.std(self.history[-l:])
    
    def my_events(self, price):
        self.events['cross_resistance'] = {k: price>i for k,i in self.levels['resistance'].items()}
        self.events['cross_support'] = {k: price<i for k,i in self.levels['support'].items()}
        
        self.events['cross ema from below'] = {k: 1 if price>=i and self.history[-1]<i 
                   else 0 for k,i in self.levels['ema'].items()}
        self.events['cross ema from above'] = {k: 1 if price<=i and self.history[-1]>i 
                   else 0 for k,i in self.levels['ema'].items()}
        self.events['price increase'] = 1 if price>self.history[-1] else 0
    
    def _flatten(self, dics):
        return np.fromiter(dics.values(), dtype = float)
    
    def vector(self):
        arr = np.array([])
        for event in self.events.values():
            try:
                arr = np.append(arr, self._flatten(event))
            except:
                arr = np.append(arr, event)
        return arr
        
    def step(self, price, levels):
        self.eda(levels)
        self.my_events(price)        
        return self.vector()
        
    def add_to_history(self, price):    
        self.history = np.append(self.history, price)
        
    def delete_history(self):
        self.history = self.history[:-1]
        
        
'''        
a = close_data.values
        
m = Metrics(a[:100])
for i in a[100:200]:
    m.eda(levels = [10,20,50])
    m.my_events(i)
    a1 = m.vector()
    
    Neural network starts from here
    
    m.step(i)
    
'''



