#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from ns3gym import ns3env
from ddqn import DDQN
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from utils.continuous_environments import Environment

duel_DQN  = 0    ### for duel network with replay experience otherwise put zero

def parse_args(args):


    
    parser = argparse.ArgumentParser(description='RL params')
    parser.add_argument('--with_PER', dest='with_per', action='store_true', help="With Replay Buffer ")
    
    if (duel_DQN == 1):
        parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling DQN ")
    else:
        parser.add_argument('--dueling', dest='dueling', action='store_false', help="Simple DQN with replaybuffer")
        
    parser.add_argument('--nb_episodes', type=int, default=200, help="Number of simulation episodes")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for replay buffer")
    parser.add_argument('--consecutive_frames', type=int, default=4, help="action repeat")
    parser.add_argument('--render', dest='render', action='store_true', help="Render env")
    parser.add_argument('--env', type=str, default='ns3-v0',help="Ns3-Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.set_defaults(render=False)
    return parser.parse_args(args)

def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env = Environment(gym.make('ns3-v0'), args.consecutive_frames)
    state_dim = env.get_state_size()
    action_dim = gym.make('ns3-v0').action_space.n
    
    algo = DDQN(action_dim, state_dim, args)
   
    if (duel_DQN == 1):
        filename = 'DuelingDQN_withreplay_Sim_simulation_Pyscript'
    else:
        filename = 'DQN_withreplay_Sim_simulation_Pyscript'
    # Train
    stats = algo.train(env, args, filename)


if __name__ == "__main__":
    main()
### can pass the arguments from command prompt