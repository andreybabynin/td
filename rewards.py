#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:21:13 2019

@author: enron07
"""
import numpy as np

class Reward:
    
    def __init__(self):
        pass
    
    def multi_reward(self, r_cash, step, action,  mode=True):
        '''Multi reward scheme'''
        if mode:
            return (1/step)*self.hate_cash() + self.holding_impatience(step, action) +\
            r_cash*np.log(step)
        else:
            return r_cash
    
    def hate_cash(self, mode = True, hate = -0.05):
        '''Penalty for staying in cash'''
        if mode:
            return hate
        else:
            return 0
    
    def holding_impatience(self, step, action, mode = True, impatience = -0.02, hold_min=6, hold_max = 30): #2,5 hours and 30 min
        '''Penalty for very long positions'''
        if mode:
            if action ==2:
                return impatience/np.log(step)
            else: return 0
        else:
            return 0
    
    def price_direction(self):
        '''Reward for guessing riht price direction and taking appropriate long/short positions''' 
        pass
    
