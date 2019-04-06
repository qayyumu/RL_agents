#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env
import random
from tqdm import tqdm
import json

# Environment initialization
# port = 5555; simTime = 10; startSim = True
# stepTime = 0.1; seed = 132; simArgs = {"--simTime": simTime}; debug = False
# env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)


### environment creation
env = gym.make('ns3-v0')
ob_space = env.observation_space
ac_space = env.action_space
s_size = ob_space.shape[0]
a_size = ac_space.n
q_table = np.zeros([s_size,a_size])
#q_table = np.random.randn(4, 4)
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)


###global parameters declaration
alpha = 0.4
gamma =0.9999
#epsilon = 0.017
epsilon = 1.0
epsilon_min = 0.017
epsilon_decay = 0.99
total_episodes = 200  ###run simulations
max_env_steps = 95  ### each simulation consist of 10sec (.1*100=10sec)
env._max_episode_steps = max_env_steps

###history for plotting purpose
time_history = [0]
rew_history = [0]

def epsilon_greedy_policy(state_, epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: q_table[(state_,x)])


def update_q_table(prev_state_, action_, reward_, nextstate_, alpha, gamma):
    qa = max([q_table[(nextstate_, a)] for a in range(env.action_space.n)])
    q_table[(prev_state_,action_)] += alpha * (reward_ + gamma * qa - q_table[(prev_state_,action_)])
    
    
filename = 'QLearning_Sim_simulation_Pyscript'
tqdm_e = tqdm(range(total_episodes), desc='Score', leave=True, unit=" episodes")
RL_data = {"RL-agent":[]}
#create a list
data_holder = RL_data["RL-agent"]

for epi in tqdm_e: #range(total_episodes):

    state = env.reset()
    #print(state)
    state = np.reshape(state, [1, s_size])
    rewardsum = 0
    for time in range(max_env_steps):

        action = epsilon_greedy_policy(state.argmax(),epsilon)
        next_state,reward, done, _ = env.step(action)
        if(reward<0):
            reward = 0;
        
        next_state = np.reshape(next_state, [1, s_size])
        #print(state,next_state,action,reward,done)
        
        update_q_table(state.argmax(), action, reward, next_state.argmax(), alpha, gamma)
        #print(q_table)
        # Finally we update the previous state as next state
        state = next_state

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(epi, total_episodes, time, rewardsum, epsilon))
            break

        rewardsum += reward
        
        if epsilon > epsilon_min: 
           epsilon *= epsilon_decay
           
    #print('Step ',time,'  Episode', epi,epsilon)
    tqdm_e.set_description("Score: " + str(rewardsum))
    tqdm_e.refresh()
    time_history.append(time)
    rew_history.append(rewardsum)
    data_holder.append({'Time':time})
    data_holder.append({'Reward':rewardsum})
    


print(time_history)
print(rew_history)

with open(filename+'.txt', 'w') as outfile:  
    json.dump(data_holder, outfile)


print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(time_history)), time_history, label='Steps', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(rew_history)), rew_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig(filename+'.pdf', bbox_inches='tight')
plt.show()
print(q_table)
