import numpy as np
from random import choice

class SimpleReplay: # need to decouple with agent object
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        
    def push(self, step, a, r):
        self.buffer.append((step, a, r))
        if len(self.buffer)>self.capacity:
            self.buffer = self.buffer[-self.capacity:]
    
    def sample_step(self): 
        t = choice(self.buffer[:-1]) #to escape choosing the last value and overcounting events
        return  t[0], t[1], t[2]

class HindsightReplay:
    '''
    https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8
    https://github.com/orrivlin/Hindsight-Experience-Replay---Bit-Flipping
    https://arxiv.org/pdf/1707.01495.pdf
    '''
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        
class PrioritizedReplay(SimpleReplay):
    '''
    https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
    https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb
    Idea:
        1) add weights to each replay according to the loss (td - difference)
        2) update these weights if use sample many times to reduce the probability of being chosen next time
    '''
    
    def __init__(self, capacity, alpha = 0.4, beta = 0.7):
        super().__init__(capacity)
        self.priority_buffer = []
        self.alpha = alpha # randomness coefficient (0 - pure random, 1 - highest priority)
        self.beta = beta
        
    ''' Override metohds of simple replay'''    
    def push(self, step, loss):
        self.buffer.append(step)
        self.priority_buffer.append(loss)
        if len(self.buffer)>self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            self.priority_buffer = self.priority_buffer[-self.capacity:]

    def normalization(self):     
        return [np.power(np.abs(loss), self.alpha)/np.sum(np.power(np.abs(self.priority_buffer[:-1]), self.alpha)) for loss in self.priority_buffer[:-1]]
     
    def sample_step(self):
        step = np.random.choice(self.buffer[:-1], 
                                p = np.asarray(self.normalization()).ravel())
        step_index = self.buffer.index(step)
        # Importance sampling weights
        self.priority_buffer[step_index] = np.power(1/(self.capacity *\
                                self.priority_buffer[step_index]), self.beta)
        return step 
