import gym
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datetime
import matplotlib.pyplot as plt

import hickle as hkl

from QNetworks import QNetwork1
from ReplayBuffer import ReplayBuffer
from Agents import *



# Initialising Environment

env = gym.make('Acrobot-v1')
env.seed(0)

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.n

# Defining DQN Algorithm

def dqn(n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    scores = []                 # list containing scores from each episode
    scores_window_printing = deque(maxlen=10) # For printing in the graph
    scores_window= deque(maxlen=100)  # last 100 scores for checking if the avg is more than 195
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores_window_printing.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")  
        if i_episode % 10 == 0: 
            scores.append(np.mean(scores_window_printing))        
        if i_episode % 100 == 0: 
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=-80.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    return [np.array(scores),i_episode-100]

# Trial run to check if algorithm runs and saves the data

no_siblings = 30
sibling_scores = []
sibling_lives = np.zeros(no_siblings)

begin_time = datetime.datetime.now()
for i in range(no_siblings):
    
    agent = Agent5(state_size=state_shape,action_size = action_shape,seed = 0)
    [temp_scores,sibling_lives[i]] = dqn()
    sibling_scores.append(temp_scores)
time_taken = datetime.datetime.now() - begin_time
# Saving the files

hkl.dump([sibling_lives,sibling_scores,time_taken],'DQN_Acrobot_5')
# load variables from filename
#a,b,c = hkl.load(filename)