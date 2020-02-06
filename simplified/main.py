import numpy as np
from settings import COMISSION, SLIPAGE, REPLAY_PROB, TRAINING, REPLAY_BUFFER_SIZE, LAYERS, DUELING, DOUBLE
from Qnet import Qnet, DuelingQnet
from replay2 import SimpleReplay
from agent2 import DQN, play_step, td_loss, update_target
from rewards2 import Reward
import torch.optim as optim 
from tensorboardX import SummaryWriter 
from env2 import Environment
from metrics2 import Metrics
import pandas as pd 
f = pd.read_csv('GAZP_180217_190217.csv', header = 0, sep =',')


if __name__ == "__main__":   
    
    engine = DuelingQnet(layers = LAYERS) if DUELING else Qnet(layers = LAYERS)
    agent = DQN(engine)
    target_agent = DQN(engine) if DOUBLE else None
    
    metrics = Metrics(f, 'support', 'resistance')
    metrics.eda()
    
    env = Environment(slipage=SLIPAGE, comission=COMISSION)
    rb = SimpleReplay(REPLAY_BUFFER_SIZE)
    rev = Reward()
    
    optimizer = optim.Adam(agent.engine.parameters())
    writer = SummaryWriter(comment = '5 epoch Double dueling Q learning no comissions gamma')
    
    loss_list = []
    reward_list = [] 

    def optimizer_step(loss):
        optimizer.zero_grad()
        loss.backward() #https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
        optimizer.step()
        
    for epoch in range(2):
        print('Epoch {} started'.format(epoch))
        for step in range(metrics._len-metrics._max):
            a = play_step(agent, step, env, metrics)
            r = rev.simple_reward(agent._step-1, env)
                    
            reward_list.append(r)
            if TRAINING:
                loss = td_loss(agent, step, a, r, target_agent, env, metrics)
                optimizer_step(loss)
                loss_list.append(loss.detach().item())
    
                if loss > np.mean(loss_list[-100:]):
                    rb.push(step, a, r)    
                ''' learning from replay '''
                if np.random.rand() < REPLAY_PROB and step > 500:      
                    step, a, r = rb.sample_step()
                    agent.qvalues = agent.get_q_values(step, env.action_space(a), metrics)
                    loss = td_loss(agent, step, a, r, target_agent, env, metrics)
                    optimizer_step(loss)
                ''' update target function for double learning'''
                if agent._step % 100 == 0:
                    if target_agent != None:
                        update_target(agent, target_agent)                  
    
                    writer.add_scalar('reward', np.mean(reward_list[-100:]), agent._step)
                    writer.add_scalar('loss', np.mean(loss_list[-100:]), agent._step)
                    print('Loss is {}'.format(np.mean(loss_list[-100:])))
                    print('Reward is {}'.format(np.mean(reward_list[-100:])))
                
    writer.close()
