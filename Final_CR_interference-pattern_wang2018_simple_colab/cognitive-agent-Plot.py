#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
Policy_value_iteration = 1     ### 1 for Value_Policy_iteration_Both benchmarking  
                               ### 2 for Value_iteration benchmarking 
                               ### 3 for Policy_iteration benchmarking 

def readJson(filename):
    
    time_history = [0]
    rew_history = [0]
    
    with open(filename+'.txt', 'r') as outfile:  
     data = json.load(outfile)
    
    for i in range(0,len(data),2):
        time_history.append(data[i]['Time']+1)
        rew_history.append(data[i+1]['Reward'])
    
    return time_history, rew_history

    

##### Read the Qlearning
filename = 'QLearning_Sim_simulation_Pyscript'
time_history_QL,rew_history_QL = readJson(filename)

##### Reading the Sarsa performance    
filename = 'SARSA_Sim_simulation_Pyscript'
time_history_SA,rew_history_SA = readJson(filename)

##### Reading the DQN_without_replay   
filename = 'DenseDQNwithoutreplay_Sim_simulation_Pyscript'
time_history_DQN_withoutReplay,rew_history_DQN_withoutReplay = readJson(filename)

##### Reading the DQN_with_replay   
filename = 'DQN_withreplay_Sim_simulation_Pyscript'
time_history_DQN_withReplay,rew_history_DQN_withReplay = readJson(filename)

##### Reading the DuelDQN_with_replay   
filename = 'DuelingDQN_withreplay_Sim_simulation_Pyscript'
time_history_Duel_DQN_withReplay,rew_history_Duel_DQN_withReplay = readJson(filename)

##### Reading the Vanilla Policy   
filename = 'VPolicyGrad_Sim_simulation_Pyscript'
time_history_Vpolicy,rew_history_VPolicy = readJson(filename)

##### Reading the PPO   
filename = 'PPO_Sim_simulation_Pyscript'
time_history_ppo,rew_history_ppo = readJson(filename)





if(Policy_value_iteration == 1):
###value+policy iteration
    df=pd.DataFrame({'x': range(0,201), 'QL': rew_history_QL, 'SA': rew_history_SA, 'DQN_wo_Rep': rew_history_DQN_withoutReplay,'DQN_w_Rep':rew_history_DQN_withReplay,'D_DQN':rew_history_Duel_DQN_withReplay,'PPO':rew_history_ppo })
elif(Policy_value_iteration==2):
####value iteration   (Qlearning, DQN, DQN with replay, Duel DQN, )
    df=pd.DataFrame({'x': range(0,201), 'QL': rew_history_QL, 'SA': rew_history_SA, 'DQN_withoutReplay': rew_history_DQN_withoutReplay,'DQN_withReplay':rew_history_DQN_withReplay,'Duel_DQN':rew_history_Duel_DQN_withReplay })
elif(Policy_value_iteration==3):
### policy iteration approach   (Vanilla Policy Gradient, Proximal Policy Optimization)
    df=pd.DataFrame({'x': range(0,201), 'Vpolicy':rew_history_VPolicy,'PPO':rew_history_ppo })

# Data all 
#df=pd.DataFrame({'x': range(0,201), 'QL': rew_history_QL, 'SA': rew_history_SA, 'DQN_withoutReplay': rew_history_DQN_withoutReplay,'DQN_withReplay':rew_history_DQN_withReplay,'Duel_DQN':rew_history_Duel_DQN_withReplay,'Vpolicy':rew_history_VPolicy,'PPO':rew_history_ppo })




mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})
plt.style.use('seaborn-paper')
# create a color palette
palette = plt.get_cmap('Set1')

majorLocator = MultipleLocator(40)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(40)
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
 
    # Find the right spot on the plot
    axx= plt.subplot(3,3, num)
 
    # plot every groups, but discreet
    for v in df.drop('x', axis=1):
        plt.plot(df['x'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
 
    if(num==5):
        plt.plot(df['x'], df[column], marker='', color=palette(7), linewidth=2.4, alpha=0.9, label=column)
    # Plot the lineplot
    else:
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)
 
    # Same limits for everybody!
    plt.xlim(0,200)
    plt.ylim(0,100)
    axx.xaxis.set_major_locator(majorLocator)
    axx.xaxis.set_major_formatter(majorFormatter)
    axx.xaxis.set_minor_locator(minorLocator)
    axx.xaxis.set_minor_formatter(majorFormatter)
 
    # Not ticks everywhere
    if num in range(7) :
        plt.tick_params(labelbottom='off')
    if num not in [1,4,7] :
        plt.tick_params(labelleft='off')
 
    # Add title
    if(num==5):
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(7) )
    else:
        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )
        
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                    wspace=0.35)
   
#plt.suptitle("Value_Policy Iteration RL", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)


if(Policy_value_iteration == 1):
###value+policy iteration
    filename1 = 'Value_Policy_Iteration-Variant'
elif(Policy_value_iteration==2):
####value iteration   (Qlearning, DQN, DQN with replay, Duel DQN, )
    filename1 = 'ValueIteration-Variant'
elif(Policy_value_iteration==3):
### policy iteration approach   (Vanilla Policy Gradient, Proximal Policy Optimization)
    filename1 = 'PolicyIteration-Variant'


plt.savefig(filename1+'.pdf', bbox_inches='tight')
plt.show()
