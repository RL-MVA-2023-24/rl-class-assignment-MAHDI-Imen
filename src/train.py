from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import random
import os
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.best_model = torch.nn.Sequential(nn.Linear(6, 64),
                                nn.ReLU(),
                                nn.Linear(64, 64),
                                nn.ReLU(), 
                                nn.Linear(64, 4))


    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return greedy_action(self.best_model, observation)

    def save(self, path):
        agent_state = {
            'state_dict': self.best_model.state_dict(),
        }

        with open(path, 'wb') as f:
            pickle.dump(agent_state, f)

    def load(self):
        # load best model state dict from pickle
        import pickle
        with open('env_5_model.pkl', 'rb') as f:
            best_model_state_dict = pickle.load(f)

        self.best_model.load_state_dict(best_model_state_dict)

#     def train(self, env):
#         max_episode = 100
#         scores = self.best_model.train(env, max_episode)



# class ReplayBuffer:
#     def __init__(self, capacity, device):
#         self.capacity = capacity # capacity of the buffer
#         self.data = []
#         self.index = 0 # index of the next cell to be filled
#         self.device = device
#     def append(self, s, a, r, s_, d):
#         if len(self.data) < self.capacity:
#             self.data.append(None)
#         self.data[self.index] = (s, a, r, s_, d)
#         self.index = (self.index + 1) % self.capacity
#     def sample(self, batch_size):
#         batch = random.sample(self.data, batch_size)
#         return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
#     def __len__(self):
#         return len(self.data)



def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

# class dqn_agent:
#     def __init__(self, config, model):
#         device = "cuda" if next(model.parameters()).is_cuda else "cpu"
#         self.gamma = config['gamma']
#         self.batch_size = config['batch_size']
#         self.nb_actions = config['nb_actions']
#         self.memory = ReplayBuffer(config['buffer_size'], device)
#         self.epsilon_max = config['epsilon_max']
#         self.epsilon_min = config['epsilon_min']
#         self.epsilon_stop = config['epsilon_decay_period']
#         self.epsilon_delay = config['epsilon_delay_decay']
#         self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
#         self.model = model 
#         self.criterion = torch.nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
#     def gradient_step(self):
#         if len(self.memory) > self.batch_size:
#             X, A, R, Y, D = self.memory.sample(self.batch_size)
#             QYmax = self.model(Y).max(1)[0].detach()
#             #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
#             update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
#             QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))

#             loss = self.criterion(QXA, update.unsqueeze(1))
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step() 
#             return loss.item()
#         return 0.0
    
#     def train(self, env, max_episode):
#         episode_return = []
#         episode = 0
#         episode_cum_reward = 0
#         state, _ = env.reset()
#         epsilon = self.epsilon_max
#         step = 0
#         loss = 0.0
#         while episode < max_episode:
#             # update epsilon
#             if step > self.epsilon_delay:
#                 epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

#             # select epsilon-greedy action
#             if np.random.rand() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = greedy_action(self.model, state)

#             # step
#             next_state, reward, done, trunc, _ = env.step(action)
#             self.memory.append(state, action, reward, next_state, done)
#             episode_cum_reward += reward

#             # train
#             loss += self.gradient_step()

#             # next transition
#             step += 1
#             if step%200 == 0:
#                 print("Episode ", '{:2d}'.format(episode+1), 
#                       ", epsilon ", '{:6.2f}'.format(epsilon), 
#                       ", batch size ", '{:5d}'.format(len(self.memory)), 
#                       ", episode return ", '{:4.1f}'.format(episode_cum_reward),
#                       ", loss ", '{:4.1f}'.format(loss/200),
#                       sep='')
#                 episode += 1
#                 loss = 0
#                 state, _ = env.reset()
#                 episode_return.append(episode_cum_reward)
#                 episode_cum_reward = 0
#             else:
#                 state = next_state

#         return episode_return