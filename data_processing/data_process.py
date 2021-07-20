import math
import os
import os.path as osp
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from itertools import permutations

import numpy as np
import itertools
import tensorflow as tf

import torch
from torch_geometric.data import Data

from features import features_description
from data_process_utils import *

file_name = f'./data/training_tfexample.tfrecord-0{9:04d}-of-01000'
dataset = tf.data.TFRecordDataset(file_name, compression_type='')


i = 0
for data in dataset.as_numpy_iterator():    
    parsed = tf.io.parse_single_example(data, features_description)
    if not if_vehicle_interaction(parsed):
        continue

    interaction_range = get_interaction_range(parsed, 2)

    origin = get_interaction_frame_origin(parsed)

    polyline_ids = get_field_polyline_ids(parsed, interaction_range)

    traj_ids = get_field_traj_ids(parsed, interaction_range)

    tl_ids = get_field_traffic_lights(parsed, interaction_range)

    graph_data = build_graphs(parsed, polyline_ids, traj_ids, 1, 1)
    
    torch.save(data, osp.join('processed_graph_data', 'data_{}.pt'.format(i)))
    i += 1