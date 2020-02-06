import numpy as np

class Metrics:
    
    def __init__(self, history, *args, training=True):
        '''
            history: df
        '''
        self._training = training
        self.history = np.array(history)
        self._len = self.history.shape[0]
        self._define_col(history)
        self.define_inst(*args)
        self.returns(1,5,10)
             
    def _define_col(self, history):
        self._close_cl = history.columns.get_loc('<CLOSE>')
        self._volume_cl = history.columns.get_loc('<VOL>')
    
    def define_inst(self, *args):
        self._instruments = []
        self._events = []
        dic = {'resistance': self._resistance, 'support': self._support}
        dic_e = {'resistance': self._resistance_break, 'support': self._support_break}
        for inst in args:
            if inst in dic.keys():
                self._instruments.append(dic[inst])
                self._events.append(dic_e[inst])
        self._n_inst = len(self._instruments)
    
    ''' Library of functions '''
    
    def _resistance(self, l, step):
        return self.history[step+self._max-l: step+self._max, self._close_cl].max()
    
    def _support(self, l, step):
        return self.history[step+self._max-l: step+self._max, self._close_cl].min()
    
    def _resistance_break(self, levels_arr, price):
        return price>levels_arr
           
    def _support_break(self, levels_arr, price):
        return price<levels_arr
    
    def returns(self, *args):
        self.ret = {}
        for n in args:
            self.ret[n] = np.array(self.history[n:, self._close_cl])/np.array(self.history[:-n, self._close_cl])-1
    
    ''' Explanatory statistics '''
    def eda(self, levels=[10,30,50], axis = 0):
        self._n_levels = len(levels)
        self._max = max(levels)      
        if self._training:
            self.events_hist = np.zeros([1, self._n_levels*self._n_inst])
            for step in range(self._len-self._max):
                price = self.history[step+self._max, self._close_cl]
                inst_event_hist = np.array([])
                for inst, event in zip(self._instruments, self._events):
                    event_arr = np.array([])
                    for l in levels:
                        event_arr = np.hstack((event_arr, event(inst(l, step), price)))
                    inst_event_hist = np.hstack((inst_event_hist, event_arr))
                self.events_hist = np.vstack((self.events_hist, inst_event_hist.reshape(1,-1)))
            self.events_hist = self.events_hist[1:,:]
        else:
            pass
