#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from ns3gym import ns3env
from tqdm import tqdm
import json



ITERATION = 200 
max_env_steps = 95
GAMMA = 0.99


def main():
    
    env = gym.make('ns3-v0')
    action_dim = env.action_space.n
    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)

    #PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA,clip_value=0.1, c_1=.8, c_2=0.1)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA,clip_value=0.3, c_1=.01, c_2=0.01)
    
    #saver = tf.train.Saver()
    time_history = []
    rew_history = []
    
    filename = 'PPO_Sim_simulation_Pyscript'
    RL_data = {"RL-agent":[]}
    #create a list
    data_holder = RL_data["RL-agent"]


    with tf.Session() as sess:
        #writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        reward = 0
        success_num = 0
        tqdm_e = tqdm(range(ITERATION), desc='Score', leave=True, unit=" episodes")

        for iteration in tqdm_e: #range(ITERATION):  # episode
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            indexx = 1
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
            
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)
                
                if(reward<0):     ### force the reward to zero for negative
                    reward = 0;
                
                indexx+=1
                if(indexx > max_env_steps):   ### if simulation steps have finished
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break;

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs
                    
                    
            time_history.append(indexx)
            rew_history.append(sum(rewards) )
            tqdm_e.set_description("Score: " + str(sum(rewards) ))
            tqdm_e.refresh()
            data_holder.append({'Time':indexx})
            data_holder.append({'Reward':sum(rewards)})

            

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])

            #summary = PPO.get_summary(obs=inp[0],
            #                          actions=inp[1],
            #                          rewards=inp[2],
            #                          v_preds_next=inp[3],
            #                          gaes=inp[4])[0]

            #writer.add_summary(summary, iteration)
        #writer.close()
        
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
        #plt.savefig('ppo_policy_learning_2.pdf', bbox_inches='tight')
        #plt.savefig('CartPole_sim_policy_learning_1.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
