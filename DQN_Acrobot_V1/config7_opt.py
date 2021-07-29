'''
# Case 5- (-Q +E +T)

### Neural Network 
Input Layer - 4 nodes (State Shape)
Hidden Layer 1 - 64 nodes
Hidden Layer 2 - 64 nodes
Output Layer - 2 nodes (Action Space)
Optimizer - zero_grad()

### Networks Update Frequency (NO)
Frequency of network switch - Every episode

###  Experience Replay (YES)
Total Replay Buffer Size - 10,000
Mini Batch Size - 64

### Loss Clipping (YES)
Gradient is clipped to 1 & -1
'''

import numpy as np
import random
from collections import namedtuple, deque



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1                 # hard update because we want full copy
LR = 5e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        '''
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        '''
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)

 
        self.learn(state, action, reward, next_state, done, GAMMA)

        # Updating the Network every step taken
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        '''
        self.t_step = (self.t_step + 1)
        ''' 
               
            


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, state, action, reward, next_state, done, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones = experiences

        state = torch.from_numpy(np.vstack([state])).float().to(device)
        action = torch.from_numpy(np.vstack([action])).long().to(device)
        reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_state = torch.from_numpy(np.vstack([next_state])).float().to(device)
        done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)

        # Get max predicted Q values (for next states) from target model

        Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state).gather(1, action)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
                  


