# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:37:36 2019

@author: andrey.babynin
"""

import pandas as pd
import numpy as np

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
        10) is_done - game is over if cum_return is -1 or portfolio_value is 0
        11) position - current dollar value of the position
        12) position_size - % size of position relative to available cash
    
    '''
    def __init__(self, slipage=0.005, comission=0.005, gamma=0.0001, endowment=10**6, position_size=0.01):
        self.slipage = slipage
        self.comission = comission
        self.step_number = 0 
        self.gamma = gamma 
        self.cum_reward = 0
        self.endowment = endowment
        self.cash = endowment
        self.action_space_next = [0,1,4] # initial state open 
        self.position = {}
        self.position_size = position_size
        self.history = {'position': [], 'actions': [], 'reward': [], 'abs': 0, 
                        'portfolio': [], 'volume': []}
        
    @staticmethod
    def position_value(volume, initial_price, share_price, type_position):
        if type_position == 'long':
            value = (share_price - initial_price)*volume
        elif type_position == 'short':
            value = (initial_price - share_price)*volume
        return value
    
    def step(self, action, share_price):
                    
        '''
        If go long/short
        '''
        if (action == 0) or (action ==1):
            self.action_space_next = [2,3] 
            
            '''
            Slipage mechanics: higher price when goes long, lower otherwise
            '''
            amount_invested = self.cash*self.position_size*(1-self.comission)
                        
            if action ==0:
                self._buy_price = share_price*(1+self.slipage)
                self._volume = amount_invested/self._buy_price
                self._type_position = 'long'
                
            else:
                self._sell_price = share_price*(1-self.slipage)
                self._volume = amount_invested/self._sell_price
                self._type_position = 'short'
            
            
            self.position = {'abs': self.position_value(self._volume, self._buy_price if self._type_position =='long'
                                                            else self._sell_price, share_price, self._type_position), 
                             'number': self._volume,
                             'type': self._type_position}
            self.cash = self.cash*(1 - self.position_size)
            
            
        elif action == 2:
            self.action_space_next = [2,3]
            '''
            Recalculate position while holding
            '''
            self.position['abs'] = self.position_value(self.position['number'],
                         self._buy_price if self._type_position =='long'
                                                            else self._sell_price,
                                                            share_price,
                                                            self._type_position)
            
        # recalculate while closing position 
        elif action == 3:
            self.action_space_next = [0,1,4]
  
            if self.position['type'] == 'long':
                self.cash = self.cash + self.position_value(self._volume, self._buy_price if self._type_position =='long'
                                                            else self._sell_price, share_price, self._type_position)*(1-self.comission)
            elif self.position['type'] == 'short':
                self.cash = self.cash + self.position_value(self._volume, self._buy_price if self._type_position =='long'
                                                            else self._sell_price, share_price, self._type_position)*(1-self.comission)
            
            self.position = {'abs': 0, 'number': 0, 'type': ''}
                 
        else:
            self.action_space_next = [1,1,0,0,1]         
        self.step_number +=1
        '''Append history'''
        self.history['position'].append(self.position['abs'])
        self.history['volume'].append(self.position['number'])
        self.history['actions'].append(action)
        self.history['reward'].append(self.reward)
        self.history['portfolio'].append(self.portfolio_value(share_price))
        self.history['steps'] = self.step_number
        
        return self.position, self.reward(self.step_number), self.is_done, self.action_space_next
    
    
    def portfolio_value(self, share_price):
        
        return self.cash + self.position_value(self._volume, self._buy_price if self._type_position =='long'
                                                            else self._sell_price, share_price, self._type_position)
    
    def reward(self, step):
        '''Reward is the difference between portfolio value between two time steps'''
        if step ==1:
            pass
        else:
            return self.history['portfolio'][-1] - self.history['portfolio'][-2]

    
    def reset(self):
        self.history = {}
        self.cash = self.endowment

    def is_done(self):
        if self.cash <=0:
            return True
        else:
            return False
    
    
'''
Simple example checking mechanics
''' 
import numpy as np
import pandas as pd


raw_data = pd.read_csv('GAZP_180217_190217.csv', header = 0, sep =';')     
 
close_data = raw_data['<CLOSE>']   
#actions = np.random.choice(5, len(close_data), p = [0.02,0.02, 0.46, 0.03,0.47])   

close_data_s = close_data.head()
        
env = Environment()   
step =0
for tick in close_data.values:
    if step == 0:
        pos, r, _, space = env.step(np.random.choice([0,1,4]), tick)
        step +=1
    else:
        step +=1
        pos, r, _, space = env.step(np.random.choice(space), tick)


portfolio = env.history['portfolio']
actions = env.history['actions']

import matplotlib.pyplot as plt


plt.clf()
_ = plt.plot(portfolio[:100])
plt.show()


_ = plt.plot(actions[:10])
plt.show()
