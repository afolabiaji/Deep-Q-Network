# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import gym
import cv2
import matplotlib.pyplot as plt
from gym import wrappers
import numpy as np

env = gym.make("LunarLander-v2")
env = wrappers.Monitor(env, "./LunarLander-v2", force=True)
# Create render fucntion to run on server
# render = lambda: plt.imshow(env.render(mode="rgb_array"))

observation = env.reset()


for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())

    if done:
        observation = env.reset()

env.close()