from settings import GAMMA
import numpy as np
import torch

class DQN:
    def __init__(self, engine, explore_start=0.3, explore_stop=0.001, 
                 decay_rate=0.000001):
        self.engine = engine # NN architecture
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self._step = 0  #global step at the training
        self._lstm = engine.__class__.__name__ == 'DuelingLSTM'
        self._dueling = engine.__class__.__name__ == 'DuelingQnet'
        
    def explore_prob(self):
        return self.explore_stop + (self.explore_start -\
                self.explore_stop)*np.exp(-self.decay_rate *self._step)
       
    def get_q_values(self, step, space, metrics = None):
        return self.dueling_q(step, space, metrics) if self._dueling else self.simple_q(step, space, metrics)
     
    def dueling_q(self, step, space, metrics = None):
        Qs = torch.empty(0)
        adv_proxy = torch.empty(0)
        for a in space:
            action_vector = np.zeros([5])
            if self._lstm:
                pass
            else: 
                action_vector[a]+=1
                v = np.append(metrics.events_hist[step, :], action_vector) 
                for n in metrics.ret.keys():
                    v = np.append(v, metrics.ret[n][step])
                s, adv = self.engine(torch.Tensor(v))
                Q_proxy = s + adv
                adv_proxy = torch.cat((adv_proxy, adv), -1)
                Qs = torch.cat((Qs, Q_proxy), -1)
        return Qs - adv_proxy.mean()
    
    def simple_q(self, step, space, metrics = None):
        Qs = torch.empty(0)
        for a in space:
            action_vector = np.zeros([5])
            if self._lstm:
                pass
            else:
                action_vector[a]+=1
                v = np.append(metrics.events_hist[step, :], action_vector) 
                for n in metrics.ret.keys():
                    v = np.append(v, metrics.ret[n][step])
                Q = self.engine(torch.Tensor(v))
                Qs = torch.cat((Qs, Q), -1)
        return Qs
    
    def get_next_q_value(self, step, action, env = None, metrics= None):
        space = env.action_space(action)
        return self.get_q_values(step, space, metrics).max()
            
'''Double Q learning update target weights '''
def update_target(current_model, target_model):
    target_model.engine.load_state_dict(current_model.engine.state_dict())

def td_loss(current_model, step, action, reward, target_model = None, env = None, metrics= None):

    if target_model != None:
        next_qvalue = target_model.get_next_q_value(step, action, env, metrics).detach()
    else:
        next_qvalue = current_model.get_next_q_value(step, action, env, metrics).detach()
            
    expected_q_value  = torch.Tensor(np.array([reward])) + next_qvalue
    loss = (current_model.qvalues - GAMMA*torch.max(expected_q_value)).pow(2).mean() 
    return loss

def play_step(agent, step, env = None, metrics = None): 

    agent._step += 1 #internal step calculator in order to compute exploration probability checked
    # Need to calculate q_values even if taking random action in order to compute td_loss
    agent.qvalues = agent.get_q_values(step, env.action_space_next, metrics)
    exp_exp_tradeoff = np.random.rand()
    explore_probability = agent.explore_prob()
    if step != metrics._len-1:
        if (explore_probability > exp_exp_tradeoff): 
            action = np.random.choice(env.action_space_next)
        else:
            action = env.action_space_next[np.argmax(agent.qvalues.detach().numpy())]
    else:
        action = max(env.action_space_next) # close any positions at the end of epoch 3 - close, 4 -do nothing
       
    env.step(action, step, metrics)
    return action
