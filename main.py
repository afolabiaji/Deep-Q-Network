from collections import deque

# ------ IMPORT GYM -----------
import gym
import matplotlib.pyplot as plt

# ------ LOCAL IMPORTS --------
from preprocessing import map_and_stack_frames
from hyperparameters import *

# Initialise the Gym
env = gym.make("ALE/Breakout-v5")

# Create render fucntion to run on server
render = lambda: plt.imshow(env.render(mode="rgb_array"))
# env.reset()

observation, info = env.reset(seed=42, return_info=True)
print(observation, type(observation), observation.shape)
render()


print(env.action_space.sample())
action = env.action_space.sample()

print(preprocess_frame(env.step(action)[0], observation))

# What will we need


# Memory
replay_memory = deque(maxlen=MEMORY_SIZE)
