import numpy as np
from hyperparameters import *
from collections import deque

from model import QNetwork

import torch
import gym 

# Initialise the Gym
env = gym.make("ALE/Breakout-v5")

#Initialize replay memory D to capacity N
replay_memory = deque(maxlen=MEMORY_SIZE)

#Initialize action-value function Q with random weights h
action_value_function = QNetwork()

#Initialize target action-value function Q^ with weights h2 5 h
target_function = QNetwork()

#set model params to be same as previous model
with torch.no_grad():
    target_function.network = action_value_function.network


# Create render fucntion to run on server
render = lambda: plt.imshow(env.render(mode="rgb_array"))
observation, info = env.reset(seed=42, return_info=True)

done = False

for episode in range(20):
    for time_step in range(MAX_TIME_STEPS):
        if not done:
            render()
            if time_step < NOOP_MAX:
                action = 0
                observation, reward, done, info = env.step(action)
                continue
            elif epsilon > np.random():
                #do preprocessing steps here.
                #we want to take 4 frames and resize them then stack them
                #then go from numpy to torch
                #then pass into action value function
                action = torch.argmax(action_value_function(observation)).item()
            else:
                action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            print(observation, reward, done, info)
        else:
            break
