#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import keras
from ns3gym import ns3env
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

env = gym.make('ns3-v0')

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n
model = keras.Sequential()
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='tanh'))    ###tried relu as well but tanh converges faster  
#model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='relu'))
model.add(keras.layers.Dense(s_size, input_shape=(s_size,), activation='tanh'))
model.add(keras.layers.Dense(a_size, activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

total_episodes = 200
max_env_steps = 95
env._max_episode_steps = max_env_steps

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999
GAMMA = .99

time_history = []
rew_history = []


filename = 'DenseDQNwithoutreplay_Sim_simulation_Pyscript'
tqdm_e = tqdm(range(total_episodes), desc='Score', leave=True, unit=" episodes")
RL_data = {"RL-agent":[]}
#create a list
data_holder = RL_data["RL-agent"]
    

for e in tqdm_e:

    state = env.reset()
    state = np.reshape(state, [1, s_size])
    rewardsum = 0
    for time in range(max_env_steps):

        # Choose action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(a_size)
        else:
            action = np.argmax(model.predict(state)[0])

        # Step
        next_state, reward, done, _ = env.step(action)

        if done:
            print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                  .format(e, total_episodes, time, rewardsum, epsilon))
            break

        next_state = np.reshape(next_state, [1, s_size])

        # Train
        target = reward
        if not done:
            target = (reward + GAMMA * np.amax(model.predict(next_state)[0]))

        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        rewardsum += reward
        if epsilon > epsilon_min: epsilon *= epsilon_decay
        
    tqdm_e.set_description("Score: " + str(rewardsum))
    tqdm_e.refresh()
    time_history.append(time)
    rew_history.append(rewardsum)
    data_holder.append({'Time':time})
    data_holder.append({'Reward':rewardsum})
    


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

#plt.savefig('learning.pdf', bbox_inches='tight')
plt.savefig(filename+'.pdf', bbox_inches='tight')
plt.show()
