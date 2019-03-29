# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:27:39 2019

@author: andrey.babynin
"""
import pandas as pd 
import numpy as np
from env import Environment
from metrics import Metrics
from Qnet import Qnet, DuelingQnet, DuelingLSTM
from utils import Utils
from rewards import Reward
import torch
import torch.optim as optim
import matplotlib.pyplot as plt   
from tensorboardX import SummaryWriter 
import logging
from time import gmtime, strftime


HISTORY_BATCH = 100 # inintial history to calculate statistics
COMISSION = 0.005 #for opening and closing position
SLIPAGE = 0.005 # deviation from the price at the trading screen
REPLAY_PROB = 0.1 # probabiliy to sample day for replay
REPLAY_BUFFER_SIZE = 1000

def close_at_day(day_step, day_length, action):
    '''Closes position at the end of the day '''
    if day_step == day_length:
        day_step = 0
        done = True
        if action==2 or action ==3:              
            action = 3        
        else:
            action = 4
            #print('day is closed')
    else:
        action = action
        done = False
    return done, day_step, action 
    
def play_replay(kind = 'simple'):
    ''' Choose replay mode ['simple, 'prioritized'] '''
    if loss > np.mean(loss_list[-100:]): #невно заданные аргменты loss и loss_list потенциальная проблема
        if kind=='simple':
            rb.push(agent._step)
        if kind=='prioritized':
            rb.push(agent._step, loss)
              
    replay_prob = np.random.rand()         
    if replay_prob<REPLAY_PROB and count_days>10: # replay after some training to escape bad states      
        step = rb.sample_step()
        rb.td_rb_loss(step)

def create_logs():
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("DQN.log", mode='w')
 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # add handler to logger object
    logger.addHandler(fh)
    # initial information   
    logger.info("Program started at {}".format(strftime("%d_%b_%Y_%H:%M:%S", gmtime())))
    logger.info('Slippage is {0:.3f}, comission is {1:.3f}'.format(SLIPAGE, COMISSION))
    return logger

class DQN:
    def __init__(self, engine, explore_start=0.3, explore_stop=0.001, 
                 decay_rate=0.000001, dueling=False, lstm=False):
        #self.env = env # portfolio calculations
        #self.metrics = metrics #transformations of prices
        self.engine = engine # NN architecture
        #self.rev = rev # reward mechanics
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self._step = 0  #global step at the training
        self.day_step = 0 #step within a day   
        self.gamma = 1 # no discounting
        self.done = False
        self.dueling = dueling
        self.lstm = lstm
        
    def explore_prob(self):
        return self.explore_stop + (self.explore_start -\
                self.explore_stop)*np.exp(-self.decay_rate *self._step)

    def get_q_values(self, price, volume, space):
        states = np.empty([1,32])
        a_h = np.empty([1])
        Qs = torch.empty(0)
        
        if self.lstm:
            if self._step>self.engine._seq_length:
                for i in range(1,self.engine._seq_length):
                    s = metrics.vector(replay=True, step = self._step-1-i)
                    states = np.vstack((states, s))
                    a_h = np.vstack((a_h, env.history['actions'][self._step-1-i]))
                
            else:
                states = np.repeat(metrics.step(price, volume), 
                                   self.engine._seq_length, 0).reshape(self.engine._seq_length,-1)
        else:
            states = metrics.step(price, volume)      
        
        if self.dueling:
            adv_proxy = torch.empty(0)
            for a in space:
                if self.lstm:
                    if self._step>self.engine._seq_length:   
                        states = np.vstack((states, metrics.step(price, volume)))[1:]
                        a = np.vstack((a_h, a))[1:]
                    else:
                        a = np.repeat(a, self.engine._seq_length, 0).reshape(self.engine._seq_length,-1)
                    #print(a.shape)
                    #print(states.shape)
                    v = np.append(states, a, 1)
                else:    
                    v = np.append(states, a)
                s, adv = self.engine(torch.Tensor(v))
                Q_proxy = s+adv
                adv_proxy = torch.cat((adv_proxy, adv), -1)
                Qs = torch.cat((Qs, Q_proxy), -1)
            Qs = Qs - adv_proxy.mean()
        else:
            for a in space:
                #v = np.append(states,a,self.env.reward(a, price)[0]])
                v = np.append(states, a)
                Q = self.engine(torch.Tensor(v))
                Qs = torch.cat((Qs, Q), -1)
            
        return Qs
    
    def get_next_q_value(self, action, price_next, volume_next):
        space, _ = env.action_space(action)
        return self.get_q_values(price_next, volume_next, space).max()


class SimpleReplay: # need to decouple with agent object
    def __init__(self, capacity, dueling=False):
        self.buffer = []
        self.capacity = capacity
        self.dueling = dueling
        
    def push(self, step):
        self.buffer.append(step)
        if len(self.buffer)>self.capacity:
            self.buffer = self.buffer[-self.capacity:]
    
    def sample_step(self):      
        return np.random.choice(self.buffer[:-1]) #to escape choosing the last value and overcounting events
    
    def get_rb_q_value(self, step):
        if agent.lstm:
            v = np.empty([1,33]) #костыль размерности
            for i in range(agent.engine._seq_length):
                v= np.vstack((v,np.append(metrics.vector(replay=True, step = step-1-i),
                     env.history['actions'][step-1-i])))
            v = v[1:]
        else:
            v = np.append(metrics.vector(replay=True, step = step-1),
                      env.history['actions'][step-1])
        if self.dueling:
            s,adv = agent.engine(torch.Tensor(v))
            return s+adv            
        else:
            return agent.engine(torch.Tensor(v))
    
    def get_next_rb_q_value(self, step):
        return self.get_rb_q_value(step)
    
    def td_rb_loss(self, step):
        Q = self.get_rb_q_value(step)
        next_Q = self.get_next_rb_q_value(step).detach()
        expected_Q = torch.Tensor(np.array(env.history['syn reward'][step-1])) +\
                             agent.gamma*next_Q*(1-env.history['done'][step-1])
        loss = (Q - expected_Q).pow(2).mean()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
    

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
    
    def __init__(self, capacity, alpha = 0.4, beta = 0.7, dueling=False):
        super().__init__(capacity, dueling)
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
        #print('sum of probs', np.sum(preprocessing.normalize(np.reshape(np.array(self.priority_buffer), (1,-1))).ravel()))
        #print(preprocessing.normalize(np.reshape(np.array(self.priority_buffer), (1,-1))).ravel())    
        
        step = np.random.choice(self.buffer[:-1], 
                                p = np.asarray(self.normalization()).ravel())
        step_index = self.buffer.index(step)
        # Importance sampling weights
        self.priority_buffer[step_index] = np.power(1/(self.capacity *\
                                self.priority_buffer[step_index]), self.beta)
        return step   
    
'''Double Q learning update weights '''
def update_target(current_model, target_model):
    target_model.engine.load_state_dict(current_model.engine.state_dict())

def td_loss(cum_loss, price_next, volume_next, action, reward,  current_model,
                target_model = None, double = False):
    if double:
        next_qvalue = target_model.get_next_q_value(action, price_next,
                                                 volume_next).detach()
    else:
        next_qvalue = current_model.get_next_q_value(action, price_next,
                                                 volume_next).detach()
                
    expected_q_value  = torch.Tensor(np.array([reward])) +\
                             current_model.gamma*next_qvalue*(1-current_model.done)
    loss = (current_model.qvalues - expected_q_value).pow(2).mean()

    
    '''Accumulation of losses, update weights periodically: every 50th step'''
    cum_loss +=loss
    if agent._step % 50==0:
        optimizer.zero_grad()
        cum_loss.backward() #https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
        optimizer.step()
        cum_loss = torch.tensor(0, dtype=torch.float)
        
    return loss, cum_loss


def play_step(price, volume, day_length): # неявно заданные объект agent
    agent.done = False
    agent._step += 1
    agent.day_step +=1
    # Need to calculate q_values even if taking random action in order to compute td_loss
    
    agent.qvalues = agent.get_q_values(price, volume, env.action_space_next)
    # Probability to choose action randomly
    exp_exp_tradeoff = np.random.rand()
    explore_probability = agent.explore_prob()
    if (explore_probability > exp_exp_tradeoff): 
            # Make a random action (exploration)
        action = np.random.choice(env.action_space_next)
    else:
            # Take the biggest Q value (= the best action)
        action = env.action_space_next[np.argmax(agent.qvalues.detach().numpy())]

    agent.done, agent.day_step, action = close_at_day(agent.day_step, 
                                                             day_length, action)
    env.history['done'].append(int(agent.done))
        # add values to history
    metrics.add_to_history(price, volume)
        # recalculate position after executing action
    env.step(action, price)
        # add syntetic reward
    reward = rev.multi_reward(env.r_cash, agent._step, 
                                       action, mode=False)
    env.history['syn reward'].append(reward)
    return action, explore_probability, reward

 
if __name__ == "__main__":
     
    logger = create_logs()
    raw_data = pd.read_csv('GAZP_180217_190217.csv', header = 0, sep =',') 
    u = Utils()        
    #agent = DQN(Qnet(layers = [33,20,15,1]))  # +1 for action
    #target_agent = DQN(Qnet(layers = [33,20,15,1]))
    
    #agent = DQN(DuelingQnet(layers = [33,20,15,1]), dueling=True)
    #target_agent = DQN(DuelingQnet(layers = [33,20,15,1]), dueling=True)
    
    agent = DQN(DuelingLSTM(33,10, lin_layers = [15,1]), dueling = True, lstm=True)
    target_agent = DQN(DuelingLSTM(33,10, lin_layers = [15,1]), dueling = True, lstm = True)
    
    rb = SimpleReplay(REPLAY_BUFFER_SIZE, dueling=True)
    #rb = PrioritizedReplay(REPLAY_BUFFER_SIZE, dueling=True)
    env = Environment(comission = COMISSION, slipage = SLIPAGE)
    metrics = Metrics(raw_data['<CLOSE>'][:HISTORY_BATCH].values,
                  raw_data['<VOL>'][:HISTORY_BATCH].values, [14,30,45])              
    rev = Reward()
        
    optimizer = optim.Adam(agent.engine.parameters()) # initialize optimizer
    writer = SummaryWriter(comment = '_Dueling_LSTM') # tensorboardX
    
    loss_list = []
    reward_list = []
    count_days = 0       
    dates = sorted(set(raw_data['<DATE>'][HISTORY_BATCH:])) 
    double = True # double Q-learning
    cum_loss = torch.tensor(0, dtype = torch.float)
    for d in dates:
        count_days+=1
        date, days_prices, days_volume, days_length = u.batch_days(raw_data[HISTORY_BATCH:], d)
        #print('Next day is :', date, 'number of observations:', days_length)
        logger.info('Next day is {0}, number of observations {1}'.format(date,days_length))
        for price, price_next, volume, volume_next in zip(days_prices[:-1].values, 
                                                          days_prices[1:].values, 
                                                          days_volume[:-1].values, 
                                                          days_volume[1:].values):
            a, pr, r = play_step(price, volume, days_length)
            reward_list.append(r)

            loss, cum_loss = td_loss(cum_loss, price_next, volume_next, a, r, agent, target_agent, 
                             double=double)

            loss_list.append(loss.detach().item())
            play_replay(kind= 'simple')
                                              
            if agent._step % 100 == 0:
                  ''' Update weights of target mode'''
                  if double:
                      update_target(agent, target_agent)                  
                  # write to tensorboard
                  writer.add_scalar('reward', np.mean(reward_list[-100:]), agent._step)
                  writer.add_scalar('loss', np.mean(loss_list[-100:]), agent._step)
                  print('Loss is {}'.format(np.mean(loss_list[-100:])))
                  print('Reward is {}'.format(np.mean(reward_list[-100:])))
                  #print('Explore probability is {}'.format(pr))
    
    writer.close()




actions = env.history['actions']
portfolio = env.history['portfolio']
holding = env.history['holding period']
open = env.history['open']
close = env.history['close']
cash = env.history['cash']
reward = env.history['reward']
syneward = env.history['syn reward']
history = metrics.close
done = env.history['done']

def plot_graph(var):
    plt.clf()
    _ = plt.plot(var)
    plt.show()

plot_graph(portfolio)
plot_graph(holding)
plot_graph(reward)
plot_graph(actions)


plt.clf()
_ = plt.plot(portfolio)
#_  = plt.plot(history[100:], c = 'g')
plt.show()
'''

'''
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
'''





