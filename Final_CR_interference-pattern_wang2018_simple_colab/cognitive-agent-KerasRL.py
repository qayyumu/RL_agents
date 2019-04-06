#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###NS_LOG="Interference-Pattern:MyGymEnv" python cognitive-agent-KerasRL_1.py 
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from ns3gym import ns3env
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.policy import BoltzmannGumbelQPolicy 
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import argparse
import json
from utils.forkerasRL_environments import forkerasRL_environment




log_filename = 'dqn_test_log.json' 
callbacks = [FileLogger(log_filename, interval=100)]

def visualize_log(filename_json):
    
    with open(filename_json, 'r') as f:
         data = json.load(f)     

    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
        
        
    episodes = data['episode']
    rewards = data['episode_reward']
    episode_time = data['nb_episode_steps']     ## time


    print("Plot Learning Performance")
    mpl.rcdefaults()
    mpl.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(figsize=(10,4))
    plt.grid(True, linestyle='--')
    plt.title('Learning Performance')
    plt.plot(range(len(episode_time)), episode_time, label='Steps', marker="^", linestyle=":")#, color='red')
    plt.plot(range(len(episodes)), rewards, label='Reward', marker="", linestyle="-")#, color='k')
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.legend(prop={'size': 12})

    plt.savefig('Keras_RL_learning.pdf', bbox_inches='tight')
    plt.show()



env_tmp = gym.make('ns3-v0')
ob_space = env_tmp.observation_space
ac_space = env_tmp.action_space
env = forkerasRL_environment(gym.make('ns3-v0'),1)

print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)
nb_actions = ac_space.n


model = Sequential()
model.add(Flatten(input_shape=(1,) + env_tmp.observation_space.shape))   ### define the observation space here
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
memory = SequentialMemory(limit=5000, window_length=1)
#policy = BoltzmannQPolicy()
policy = BoltzmannGumbelQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-4, policy=policy)   ### as input we provide the action space and                

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])


#### for stochastic simulation running
dqn.fit(env, nb_steps=15300, callbacks=callbacks, visualize=False, verbose=2, nb_max_start_steps=1,nb_max_episode_steps=96)
####dqn.save_weights('stochastic_sim-kerasmodel', overwrite=True)
#####dqn.load_weights('stochastic_sim-kerasmodel')

    # Finally, evaluate our algorithm for 10 episodes.
#dqn.test(env,callbacks=callbacks, nb_episodes=10,visualize=False,verbose=2,nb_max_episode_steps=90) #,max_start_steps=1) #,nb_max_episode_steps=90, verbose=2)

visualize_log(log_filename)
