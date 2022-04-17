from hyperparameters import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSProp
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPool1D,
    Dropout,
    BatchNormalization,
    Dropout,
)

import numpy as np


# define model layers
input_shape = (MINIBATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, HISTORY_LENGTH)

test_input = np.random(size=input_shape)
# #create model
Q_NETWORK = Sequential(
    [
        Conv2D(filters=32, kernel_size=[8, 8], stride=4, activation="relu"),
        Conv2D(filters=64, kernel_size=[4, 4], stride=2, activation="relu"),
        Conv2D(filters=64, kernel_size=[3, 3], stride=1, activation="relu"),
        Flatten(),
        Dense(4, activation="softmax"),
    ]
)

Q_NETWORK.compile(
    optimizer=RMSProp(learning_rate=2.5e-4, momentum=0.95),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)
