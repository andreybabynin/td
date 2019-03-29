# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:37:36 2019

@author: andrey.babynin
"""

#import pandas as pd
#import numpy as np
import logging

class Environment:
    '''
    Parameters:
        1) slipage - deviation from current price while executing (%)
        2) comission - comission per trade (%)
        3) endowment - initial amount of money
        4) step_number - number of steps in current game
        5) gamma - discount factor
        6) action_space_next - available spaces for trader during next period:
            0/1 - unavailable/available
            action[long, short, hold, close, cash] - list of actions
        7) reward[abs, rel] - reward absolute  ($) and relative (return %) during one game
        8) cum_reward - cumulative reward per game 
        11) position - current dollar value of the position
        12) position_size - % size of position relative to available cash
    
    '''
    def __init__(self, slipage=0.005, comission=0.005, gamma=0.0001, endowment=10**6, position_size=0.01):
        self.slipage = slipage
        self.comission = comission
        self.step_number = 0 
        self.gamma = gamma 
        self.endowment = endowment
        self.cash = endowment
        self.action_space_next = [0,1,4] # initial state open 
        self.action_mask = [1,1,0,0,1]
        self.position = {'abs': 0, 'volume': 0, 'type': '', 'invested':0, 'initial': 0 }
        self.position_size = position_size
        self.history = {'position': [], 'actions': [], 'reward': [], 'syn reward': [],
                        'portfolio': [], 'volume': [], 'steps': [], 'done': [],
                        'cash': [], 'holding period': [], 'open': [], 'close': []}
        logger = logging.getLogger('agent.env')
        logger.info('Endowment is {}'.format(endowment))
        
        
    @staticmethod
    def position_value(volume, initial_price, share_price, type_position):
        return (share_price - initial_price)*volume if type_position == 'long' else (initial_price - share_price)*volume
    
    @staticmethod
    def price_with_com(price, slipage, position, close=False):
        if close==True:
            if position=='long':
                actual_price = (1-slipage)*price
            else:
                actual_price = (1+slipage)*price
        else:
            if position=='long':
                actual_price = (1+slipage)*price
            else:
                actual_price = (1-slipage)*price
        return actual_price

    def step(self, action, share_price):
        
        self.action_space_next, self.action_mask = self.action_space(action)   
        self.r_cash = self.reward(action, share_price)  
                 
        '''
        If go long/short
        '''
        if (action == 0) or (action ==1):
            
            '''
            Slipage mechanics: higher price when goes long, lower otherwise
            '''
            self.position['type'] = 'long' if action==0 else 'short'
            #print(self.position['type'])
            self.position['invested'] = self.cash*self.position_size*(1-self.comission)
            self.position['initial'] = self.price_with_com(share_price, self.slipage, 
                         self.position['type'])           
                        
            self.position['volume'] = self.position['invested']/self.position['initial']
            
            self.position['abs'] = self.position_value(self.position['volume'], 
                         self.position['initial'], share_price, self.position['type'])+self.position['invested'] 
                         
            self.cash = self.cash*(1 - self.position_size)
            
            
        elif action == 2:
            '''
            Recalculate position while holding
            '''
            self.position['abs'] = self.position_value(self.position['volume'], 
                         self.position['initial'], share_price, self.position['type']) +self.position['invested']
            
        # recalculate while closing position 
        elif action == 3:
            self.cash = self.cash + (self.r_cash+1)/(1-self.comission)*self.position['invested']

            self.position = {'abs': 0, 'volume': 0, 'type': '', 'invested':0, 'initial':0}
        # while holding cash                
        self.step_number +=1
        '''Append history'''
        self.history['position'].append(self.position['abs'])
        self.history['volume'].append(self.position['volume'])
        self.history['actions'].append(action)
        self.history['holding period'].append(self.holding_period())      
        self.history['reward'].append(self.r_cash)
        #self.history['syn reward'].append(self.r)
        self.history['portfolio'].append(self.portfolio_value())
        self.history['steps'] = self.step_number
        self.history['cash'].append(self.cash)
        
        return self.position, self.r_cash, self.action_space_next
    
    def portfolio_value(self):
        return self.cash + self.position['abs']
    
    def holding_period(self):
        if self.history['actions'][-1] in (0,1):
            self.history['open'].append(len(self.history['actions']))
        if self.history['actions'][-1] == 3:
            self.history['close'].append(len(self.history['actions']))
            return self.history['close'][-1] - self.history['open'][-1]
        return 0   
    
    @staticmethod
    def action_space(action):
        action_space_n = [2,3] if action<=2 else [0,1,4]
        action_mask_n = [0,0,1,1,0] if action<=2 else [1,1,0,0,1]
        return action_space_n, action_mask_n
            
    
    def reward(self, action, share_price):
        '''Reward is the difference between portfolio value between two time steps'''
        if action==3:
            reward_cash = ((self.position_value(self.position['volume'], 
                         self.position['initial'], self.price_with_com(share_price,
                                      self.slipage, self.position['type'], close=True), 
                         self.position['type']) +\
                        self.position['invested'])*(1-\
                        self.comission))/(self.position['invested']/(1-self.comission))-1
            
        #elif action ==4:
        #    reward = -0.005 # hate cash
        #    reward_cash = 0
        #elif action ==2:
        #    reward = 0 # take more actions
        #    reward_cash = 0
        else:
            #reward = 0
            reward_cash = 0
        return reward_cash

    def reset(self):
        self.history = {}
        self.cash = self.endowment
    