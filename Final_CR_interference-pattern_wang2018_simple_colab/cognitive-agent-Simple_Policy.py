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

from tqdm import tqdm
import json


env = gym.make('ns3-v0')
#env = gym.make('CartPole-v0')

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

s_size = ob_space.shape[0]
a_size = ac_space.n

gamma = 0.99

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.tanh)
        hidden1 = slim.fully_connected(hidden,h_size,biases_initializer=None,activation_fn=tf.nn.tanh)
        self.output = slim.fully_connected(hidden1,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))



#epsilon = 1.0               # exploration rate
#epsilon_min = 0.01
#epsilon_decay = 0.999
total_episodes = 200 #Set total number of episodes to train agent on.
max_env_steps = 95
update_frequency = 6
hidden_size = 16;

time_history = []
rew_history = []

tf.reset_default_graph() #Clear the Tensorflow graph.
myAgent = agent(lr=1e-2,s_size=s_size,a_size=a_size,h_size=hidden_size) #Load the agent.
init = tf.global_variables_initializer()

filename = 'VPolicyGrad_Sim_simulation_Pyscript'
RL_data = {"RL-agent":[]}
#create a list
data_holder = RL_data["RL-agent"]

with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []
        
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    tqdm_e = tqdm(range(total_episodes), desc='Score', leave=True, unit=" episodes")
    
    for e in tqdm_e: #range(total_episodes):
    

        s = env.reset()

        running_reward = 0
        ep_history = []
        for j in range(max_env_steps):
            
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            s1,r,d,_ = env.step(a) #Get our reward for taking an action 
            
            if(r<0):      ### force negative reward to no reward to decide bw policy
                r=0;
            
            
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            
            if (d == True):

                #print("Policy network ....saving")
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)   ##calculate the gradients
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict) 
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                    #print("Policy network will be updated")
                total_reward.append(running_reward)
                #total_lenght.append(j)
                break

        i += 1
        #print("episode: {}, time: {}, rew: {}, totreward{},".format(e,j, running_reward,np.mean(total_reward[-100:])))      
        time_history.append(j)
        rew_history.append(running_reward)
        tqdm_e.set_description("Score: " + str(running_reward))
        tqdm_e.refresh()
        #time_history.append(time)
        #rew_history.append(rewardsum)
        data_holder.append({'Time':j})
        data_holder.append({'Reward':running_reward})



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