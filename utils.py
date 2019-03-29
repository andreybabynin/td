# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:44:14 2019

@author: andrey.babynin
"""

import pandas as pd
import numpy as np

class Utils:
    def __init__(self):
        pass
    @staticmethod
    def batch_days(x, d):
        """ Split the entire history into batches by days"""
        return d, x['<CLOSE>'].loc[x['<DATE>']==d], x['<VOL>'].loc[x['<DATE>']==d], len(x.loc[x['<DATE>']==d])-1
    #-1 because price next eats one time shift
   
    @staticmethod
    def extract_time(x):
        date = datetime.datetime.strptime(str(x['<DATE>'])[:8], '%Y%m%d')
        time = datetime.datetime.strptime(str(x['<TIME>'])[:6], '%H%M%S')
        return datetime.datetime.combine(date.date(), time.time())
    
    



