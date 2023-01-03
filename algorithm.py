# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import numpy as np
import torch
from torch import nn
from collections import deque
import gym
import matplotlib.pyplot as plt
from cv2 import cv2
import random
from gym import wrappers

from hyperparameters import *
from model import QNetwork
from preprocessing import preprocess_frames, map_and_stack_frames
from torch.utils.tensorboard import SummaryWriter

#create writer for tensorboard
writer = SummaryWriter()

# get gpu if gpu exists
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Initialise the Gym
env = gym.make("ALE/Breakout-v5", render_mode='human')
env = wrappers.Monitor(env, "./Breakout-v5", force=True)
# Initialize replay memory D to capacity N
replay_memory = deque(maxlen=MEMORY_SIZE)

# Initialize action-value function Q with random weights h
action_value_function = QNetwork().to(device)

# Initialize target action-value function Q^ with weights h2 5 h
target_function = QNetwork().to(device)

# set model params to be same as previous model
with torch.no_grad():
    target_function.network = action_value_function.network


total_reward = 0
total_steps = 0

for episode in range(10000):
    # set done as false, done tells us if episode is finished
    done = False

    # reset environment
    observation = env.reset()

    # set initial epsilon to be 1
    epsilon = MAX_EPSILON

    # initialise unprocessed frames as zeros
    last_unprocessed_frames = deque(maxlen=2)
    last_unprocessed_frames.append(np.zeros([210, 160, 3]))
    last_unprocessed_frames.append(np.zeros([210, 160, 3]))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        f"./videos/episode_{episode}.mp4", fourcc, 60, (160, 210)
    )

    # initialise processed frames as zeros
    last_four_processed_frames = deque(maxlen=4)
    for _ in range(4):
        last_four_processed_frames.append(np.zeros([IMG_WIDTH, IMG_HEIGHT]))

    mapped_frames = map_and_stack_frames(
        last_four_processed_frames[0],
        last_four_processed_frames[1],
        last_four_processed_frames[2],
        last_four_processed_frames[3],
    )

    for time_step in range(1000000):
        if not done:
            # render the environment
            # env.render()
            #start the game by firing
            if time_step == 0:
                action = 1
            # check if time has passed for agent to do nothing
            # elif time_step < NOOP_MAX:
            #     action = 0
            # check if epsilon is greater than exploration
            elif epsilon < np.random.rand():
                # initialise network input as current mapped frames
                network_input = (
                    torch.tensor(mapped_frames.astype(np.float32), requires_grad=True)
                    .reshape(1, 4, 84, 84)
                    .to(device)
                )

                # take argmax of action value function to generate action
                action = torch.argmax(
                    action_value_function.forward(network_input)
                ).item()

            else:
                # sample random action if epsilon is under random number
                action = env.action_space.sample()

            print(f'Action: {action}')
            # get observation from action
            observation, reward, done, info = env.step(action)

            total_reward += reward

            last_unprocessed_frames.append(observation)

            current_processed_frame = preprocess_frames(
                last_unprocessed_frames[0], last_unprocessed_frames[1]
            )

            last_four_processed_frames.append(current_processed_frame)

            prev_mapped_frames = mapped_frames

            mapped_frames = map_and_stack_frames(
                last_four_processed_frames[0],
                last_four_processed_frames[1],
                last_four_processed_frames[2],
                last_four_processed_frames[3],
            )

            transition = {
                "mapped_frames": mapped_frames,
                "action": action,
                "reward": reward,
                "done": done,
                "prev_mapped_frames": prev_mapped_frames,
            }


            replay_memory.append(transition)

            if (total_steps > REPLAY_START_SIZE):

                # sample from replay memory for batches of experience
                if len(replay_memory) < MINIBATCH_SIZE:
                    memory_samples = random.sample(replay_memory, len(replay_memory))
                else:
                    memory_samples = random.sample(replay_memory, MINIBATCH_SIZE)

                # perform gradient descent on sample
                loss = nn.MSELoss()
                optimizer = torch.optim.RMSprop(
                    action_value_function.parameters(),
                    lr=LEARNING_RATE,
                    momentum=GRADIENT_MOMENTUM,
                )

                preds_q = []
                target_q = []

                for sample in memory_samples:

                    target_network_input = (
                        torch.tensor(sample["mapped_frames"].astype(np.float32))
                        .reshape(1, 4, 84, 84)
                        .to(device)
                    )

                    action_value_input = (
                        torch.tensor(
                            sample["prev_mapped_frames"].astype(np.float32),
                            requires_grad=True,
                        )
                        .reshape(1, 4, 84, 84)
                        .to(device)
                    )

                    if sample["done"] == True:
                        target_q.append(sample["reward"])
                    else:
                        target_q.append(sample["reward"] + GAMMA * torch.max(
                            target_function.forward(target_network_input).reshape(4)
                        ))
                    
                    preds_q.append(action_value_function.forward(action_value_input).reshape(
                        4
                    )[sample["action"]])

                # Backpropagation
                optimizer.zero_grad()
                
                target_q = torch.tensor(target_q)
                preds_q = torch.tensor(preds_q, requires_grad=True)
                output = loss(preds_q, target_q)
                print(f"Training Loss: {output}.")
                output.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', output, time_step)

                if (total_steps % TARGET_NETWORK_UPDATE_FREQ) == 0:
                    # set model params to be same as previous model
                    with torch.no_grad():
                        target_function.network = action_value_function.network

                if epsilon > MIN_EPSILON:
                    epsilon = epsilon - DECAY_RATE

            print(f"Step complete. Step: {time_step}, Episode: {episode}, Total Reward: {total_reward}")
            
            total_steps += 1
        else:
            video.release()
            break

env.close()