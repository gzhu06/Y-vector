import numpy as np
import logging
import os
import re
from glob import glob
import json
import random
import pickle

SR = 16000
SEG_LEN = 3.9

def clipped_feature(x, seg_len=SEG_LEN):
    frame_length = int(seg_len * SR)
    if x.shape[-1] > frame_length:
        bias = np.random.randint(0, x.shape[1] - frame_length)
        clipped_x = x[:, bias: frame_length + bias]
    else:
        clipped_x = x

    return clipped_x

def array2pickle(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def pickle2array(filename):
    f = open(filename, 'rb')
    array = pickle.load(f)
    f.close()
    return array