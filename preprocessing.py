from typing import Tuple

from hyperparameters import *

import numpy as np
from numpy import ndarray
from cv2 import cv2


def preprocess_frame(frame: ndarray, prev_frame: ndarray) -> ndarray:
    # First, to encode a singleframe we take themaximum value for each pixel colour
    # value over the frame being encoded and the previous frame.
    max_frame = np.maximum(frame, prev_frame)

    # Second, we then extract
    # the Y channel, also known as luminance, from the RGB frame
    img_yuv = cv2.cvtColor(max_frame, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    # and rescale it to 84 x 84.
    rescaled_frame = np.resize(y, (IMG_WIDTH, IMG_HEIGHT))

    return rescaled_frame


def map_and_stack_frames(frames: Tuple[ndarray]) -> ndarray:
    # The function w from algorithm 1 described below applies this preprocessing to the m most recent frames and stacks them to produce the input to the
    # Q-function, in whichm 5 4, although the algorithm is robust to different values of
    # m (for example, 3 or 5).
    preporcessed_frames = map(preprocess_frame, frames)
    stacked_frames = np.stack(preporcessed_frames)

    return stacked_frames
