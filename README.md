# Deep Q-Network

This is my implementation of the Deep Q Network in this seminal paper: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

## Overview

In this code I train a DQN on breakout, using OpenAI's Atari Gym environment. The doc's are here: https://www.gymlibrary.ml/, not here: 
https://gym.openai.com (these are outdated).

### Using a server

A server doesn't have a screen so 

    ```
    env.render()
    ```

won't work. You must use:
```
    render = lambda: plt.imshow(env.render(mode="rgb_array"))
```
to convert to numpy array.

## Project Structure

### Hyperparameters

Contains hyperparameters for algorithm, Q-network and preprocessing

### Preprocessing

Preprocesses the Atari images to luminescence and stacks succesive images to pass as input to network

### Model

Creates the model of Q-Network.

### Algorithm - Training

Trains the q-network on environment.

