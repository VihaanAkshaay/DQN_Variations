from QNetworks import *
from ReplayBuffer import *
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 20        # how often to update the network (When Q target is present)


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 1:    +Q +E +T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class Agent1():
    """
    CASE 1 -
    +Q +E +T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork1(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ +Q TARGETS PRESENT """
        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

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

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
                  

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 2:    +Q +E -T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent2():
    """
    CASE 2 -
    +Q +E -T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork1(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ +Q TARGETS PRESENT """
        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

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

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION ABSENT """
        #for param in self.qnetwork_local.parameters():
        #    param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 3:    +Q -E +T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent3():
    """
    CASE 3 -
    +Q -E +T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork1(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        #Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
        self.learn(state, action, reward, next_state, done, GAMMA)


        """ +Q TARGETS PRESENT """

        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.):

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

        """ -E EXPERIENCE REPLAY ABSENT """
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
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 4:    +Q -E -T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent4():
    """
    CASE 4 -
    +Q -E -T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork1(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        #Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
        self.learn(state, action, reward, next_state, done, GAMMA)


        """ +Q TARGETS PRESENT """

        # Updating the Network every 'UPDATE_EVERY' steps taken       
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())


    def act(self, state, eps=0.):

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

        """ -E EXPERIENCE REPLAY ABSENT """
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
        """ -T TRUNCATION PRESENT """
        #for param in self.qnetwork_local.parameters():
        #    param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()


""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 5:    +Q +E +T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """


class Agent5():
    """
    CASE 5 -
    -Q +E +T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ +Q TARGETS ABSENT """

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
                  

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 6:    -Q +E -T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent6():
    """
    CASE 6 -
    -Q +E -T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        """ -Q TARGETS ABSENT """


    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ -T TRUNCATION ABSENT """
        #for param in self.qnetwork_local.parameters():
        #    param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 7:    -Q -E +T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent7():
    """
    CASE 7 -
    -Q -E +T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        #Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
        self.learn(state, action, reward, next_state, done, GAMMA)


        """ -Q TARGETS ABSENT """

    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, state, action, reward, next_state, done, gamma):

        """ -E EXPERIENCE REPLAY ABSENT """
        state = torch.from_numpy(np.vstack([state])).float().to(device)
        action = torch.from_numpy(np.vstack([action])).long().to(device)
        reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_state = torch.from_numpy(np.vstack([next_state])).float().to(device)
        done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)

        # Get max predicted Q values (for next states) from target model

        Q_targets_next = self.qnetwork(next_state).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))

        # Get expected Q values from local model
        Q_expected = self.qnetwork(state).gather(1, action)
 
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ +T TRUNCATION PRESENT """
        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()

""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     CASE 8:    -Q -E -T     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

class Agent8():
    """
    CASE 8 -
    -Q -E -T
    """

    def __init__(self, state_size, action_size, seed):

        # Agent Environment Interaction
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork1(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        #Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
        self.learn(state, action, reward, next_state, done, GAMMA)


        """ -Q TARGETS ABSENT """



    def act(self, state, eps=0.):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, state, action, reward, next_state, done, gamma):

        """ -E EXPERIENCE REPLAY ABSENT """
        state = torch.from_numpy(np.vstack([state])).float().to(device)
        action = torch.from_numpy(np.vstack([action])).long().to(device)
        reward = torch.from_numpy(np.vstack([reward])).float().to(device)
        next_state = torch.from_numpy(np.vstack([next_state])).float().to(device)
        done = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)

        # Get max predicted Q values (for next states) from target model

        Q_targets_next = self.qnetwork(next_state).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        # Get expected Q values from local model
        Q_expected = self.qnetwork(state).gather(1, action)
 
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        #Gradiant Clipping
        """ -T TRUNCATION PRESENT """
        #for param in self.qnetwork_local.parameters():
        #    param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()


