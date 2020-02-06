class Environment:
    '''
    Parameters:
        1) slipage - deviation from current price while executing (%)
        2) comission - comission per trade (%)
        3) endowment - initial amount of money
        6) action_space_next - available spaces for trader during next period:
            0/1 - unavailable/available
            action[long, short, hold, close, cash] - list of actions
        7) reward - reward %
        11) position - current dollar value of the position
        12) position_size - % size of position relative to available cash
    
    '''
    def __init__(self, slipage=0.005, comission=0.005, endowment=10**6, position_size=0.01):
        self.slipage = slipage
        self.comission = comission
        self.endowment = endowment
        self.cash = endowment
        self.action_space_next = [0,1,4] # initial state open 
        self.position = {'abs': 0, 'volume': 0, 'type': '', 'invested':0, 'initial': 0 }
        self.position_size = position_size
        self.history = {'position': [], 'actions': [], 'reward': [],
                        'portfolio': [], 'volume': [],
                        'cash': [], 'holding period': [], 'open': [], 'close': []}

        
    def step(self, action, step, metrics=None):
        '''
        step - number of step
        actions:
            0 - long
            1 - short
            2 -hold
            3 - close
            4 - nothing
        '''

        share_price = metrics.history[step+metrics._max, metrics._close_cl]
        self.action_space_next = self.action_space(action)
        self.r_cash = self.reward(action, share_price)  
        '''
        If go long/short
        '''
        if (action == 0) or (action ==1):
            
            '''
            Slipage mechanics: higher price when goes long, lower when go short
            '''
            self.position['type'] = 'long' if action==0 else 'short'

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
            
        elif action == 3:
            self.cash = self.cash + (self.r_cash+1)/(1-self.comission)*self.position['invested']

            self.position = {'abs': 0, 'volume': 0, 'type': '', 'invested':0, 'initial':0}
              
        self.history['position'].append(self.position['abs'])
        self.history['volume'].append(self.position['volume'])
        self.history['actions'].append(action)
        self.history['holding period'].append(self.holding_period())      
        self.history['reward'].append(self.r_cash)
        self.history['portfolio'].append(self.cash + self.position['abs'])
        self.history['cash'].append(self.cash)
    
    def holding_period(self):
        if self.history['actions'][-1] in (0,1):
            self.history['open'].append(len(self.history['actions']))
        if self.history['actions'][-1] == 3:
            self.history['close'].append(len(self.history['actions']))
            return self.history['close'][-1] - self.history['open'][-1]
        return 0   
    
    def reward(self, action, share_price):
        '''Reward %'''
        if action==3:
            reward_cash = ((self.position_value(self.position['volume'], self.position['initial'], 
                                    self.price_with_com(share_price, self.slipage, self.position['type'], close=True), 
                                    self.position['type']) + self.position['invested'])*(1-\
                        self.comission))/(self.position['invested']/(1-self.comission))-1            
        else:
            reward_cash = 0
        return reward_cash
   
    @staticmethod
    def position_value(volume, initial_price, share_price, type_position):
        return (share_price - initial_price)*volume if type_position == 'long' else (initial_price - share_price)*volume
    
    @staticmethod
    def price_with_com(price, slipage, position, close=False):
        if close==True:
            return (1-slipage)*price if position=='long' else (1+slipage)*price
        else:
            return (1+slipage)*price if position=='long' else (1-slipage)*price
    @staticmethod
    def action_space(action):
        return [2,3] if action<=2 else [0,1,4]
