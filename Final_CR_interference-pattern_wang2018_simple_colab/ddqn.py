import sys
import random
import numpy as np

from tqdm import tqdm
from dqnagent import Agent
from random import random, randrange
import matplotlib as mpl
import matplotlib.pyplot as plt
import json


from utils.memory_buffer import MemoryBuffer
#from utils.networks import tfSummary
#from utils.stats import gather_stats

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.buffer_size = 20000
        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay


    def train(self, env, args, filename):
        """ Main DDQN Training Algorithm
        """

        results = []
        time_history = [0]
        rew_history = [0]
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        max_env_steps = 95
        
        RL_data = {"RL-agent":[]}
        #create a list
        data_holder = RL_data["RL-agent"]
        
        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()
            indexx = 1;
            
            while not done:
               
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                
                #if(r<0):
                #    r=0;
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()
                    
                indexx+=1
                if(indexx > max_env_steps):
                    break;    

            # Gather stats every episode for plotting
            #if(args.gather_stats):
            #    mean, stdev = gather_stats(self, env)
            #    results.append([e, mean, stdev])

            # Export results for Tensorboard
            #score = tfSummary('score', cumul_reward)
            #summary_writer.add_summary(score, global_step=e)
            #summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            time_history.append(indexx)
            rew_history.append(cumul_reward)
            data_holder.append({'Time':indexx-1})
            data_holder.append({'Reward':cumul_reward})

        
        
        
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

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)
