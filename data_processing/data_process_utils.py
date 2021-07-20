import math
import os
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


# a basic square range
def get_interaction_range(parsed, scale):
    ia_idx = tf.where((parsed['state/objects_of_interest']==1)).numpy()
    interaction_states = get_complete_traj(parsed, ia_idx)
    
    all_x = interaction_states[..., 0]
    all_y = interaction_states[..., 1]
    
    center_x = (np.max(all_x) + np.min(all_x)) / 2
    center_y = (np.max(all_y) + np.min(all_y)) / 2
    
    range_x = np.ptp(all_x)
    range_y = np.ptp(all_y)
    
    range = max(range_x, range_y)
    
    x_min = center_x - range * scale / 2
    x_max = center_x + range * scale / 2 
    y_min = center_y - range * scale / 2
    y_max = center_y + range * scale / 2
    
    return [x_min, x_max, y_min, y_max]
    
def get_interaction_frame_origin(parsed):
    ia_idx = tf.where((parsed['state/objects_of_interest']==1)).numpy()
    interaction_states = get_complete_traj(parsed, ia_idx)
    
    origin_x = (interaction_states[0, 10, 0] + interaction_states[1, 10, 0]) / 2
    origin_y = (interaction_states[0, 10, 1] + interaction_states[1, 10, 1]) / 2
    
    return [origin_x, origin_y]
    
# get complete trajectories, return shape [number_agents, 91, 2]
def get_complete_traj(parsed, index):
    past_states = tf.stack([tf.gather_nd(parsed['state/past/x'], indices=index), tf.gather_nd(parsed['state/past/y'], indices=index)], -1).numpy()
    current_states = tf.stack([tf.gather_nd(parsed['state/current/x'], indices=index), tf.gather_nd(parsed['state/current/y'], indices=index)], -1).numpy() 
    future_states = tf.stack([tf.gather_nd(parsed['state/future/x'], indices=index), tf.gather_nd(parsed['state/future/y'], indices=index)], -1).numpy()   
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    return all_states
    
# get past and current trajectories, return shape [number_agents, 11, 2]
def get_past_current_traj(parsed, index):
    past_states = tf.stack([tf.gather_nd(parsed['state/past/x'], indices=index), tf.gather_nd(parsed['state/past/y'], indices=index)], -1).numpy()
    current_states = tf.stack([tf.gather_nd(parsed['state/current/x'], indices=index), tf.gather_nd(parsed['state/current/y'], indices=index)], -1).numpy()   
    all_states = np.concatenate([past_states, current_states], 1)

    return all_states    

# def get_field_polyline_ids(parsed, field_range):
#     map_states = parsed['roadgraph_samples/xyz']
#     map_ids = parsed['roadgraph_samples/id']
#     num_pts = map_states.shape[0]
#     polyline_ids = []
#     for i in range(num_pts):
#         if map_states[i][0] >= field_range[0] and map_states[i][0] <= field_range[1] and map_states[i][1] >= field_range[2] and map_states[i][1] <= field_range[3]:
#             polyline_ids.append(map_ids[i][0].numpy())
#     polyline_ids = list(set(polyline_ids))
    
#     return polyline_ids

def get_field_polyline_ids(parsed, field_range):
    map_mask = parsed['roadgraph_samples/valid'][:,0].numpy() > 0
    map_states = parsed['roadgraph_samples/xyz']
    map_ids = parsed['roadgraph_samples/id']
    limit1 = map_states[:,0].numpy() >= field_range[0]
    limit2 = map_states[:,0].numpy() <= field_range[1]
    limit3 = map_states[:,1].numpy() >= field_range[2]
    limit4 = map_states[:,1].numpy() <= field_range[3]
    mask = np.all([limit1, limit2, limit3, limit4, map_mask], axis=0)
#     print(map_ids[mask].shape)
    polyline_ids = np.unique(map_ids[mask])
    
    return polyline_ids
    
def get_field_traj_ids(parsed, field_range):
    # [128, 11]
    agent_mask = tf.concat([parsed['state/past/valid'], parsed['state/current/valid']], 1).numpy() > 0
    # [128, 11]
    traj_x = tf.concat([parsed['state/past/x'], parsed['state/current/x']], 1).numpy()
    traj_y = tf.concat([parsed['state/past/y'], parsed['state/current/y']], 1).numpy()
    
    limit1 = traj_x >= field_range[0]
    limit2 = traj_x <= field_range[1]
    limit3 = traj_y >= field_range[2]
    limit4 = traj_y <= field_range[3]
    
    traj_mask = np.any(np.all([limit1, limit2, limit3, limit4, agent_mask], axis=0), axis=1)
    
    return np.nonzero(traj_mask)[0]

    
def get_field_traffic_lights(parsed, field_range):
    # [16, 11]
    tl_mask = tf.concat([parsed['traffic_light_state/past/valid'], parsed['traffic_light_state/current/valid']], 0).numpy().transpose() > 0
    # [16, 11]
    tl_x = tf.concat([parsed['traffic_light_state/past/x'], parsed['traffic_light_state/current/x']],  0).numpy().transpose()
    tl_y = tf.concat([parsed['traffic_light_state/past/y'], parsed['traffic_light_state/current/y']],  0).numpy().transpose()
    
    limit1 = tl_x >= field_range[0]
    limit2 = tl_x <= field_range[1]
    limit3 = tl_y >= field_range[2]
    limit4 = tl_y <= field_range[3]
    
    new_tl_mask = np.any(np.all([limit1, limit2, limit3, limit4, tl_mask], axis=0), axis=1)
    
    return np.nonzero(new_tl_mask)[0]


def build_graphs(parsed, map_polyline_ids, trajectory_ids, space_interval, time_interval):
    node_id = 0
    node_list = []
    edge_index_list = []
    origin = np.array(get_interaction_frame_origin(parsed))
    for i, mp_id in enumerate(map_polyline_ids):
        map_mask = parsed['roadgraph_samples/id'][:,0].numpy() == mp_id
        map_pos = parsed['roadgraph_samples/xyz'][map_mask]
        map_type = parsed['roadgraph_samples/type'][map_mask]
        
        subgraph_node_idx = 0
        for j in range(0, map_pos.shape[0], space_interval):
            if j == map_pos.shape[0] - 1:
                break
            else:
                start = map_pos[j][0:2] - origin
                if j > map_pos.shape[0] - 1 - space_interval:
                    end = map_pos[map_pos.shape[0] - 1][0:2] -origin
                else:
                    end = map_pos[j+space_interval][0:2] -origin
            m_type = map_type[j]
            node = np.concatenate([start, end, m_type, [i]])
            node_list.append(node)
            
            subgraph_node_idx += 1
            node_id += 1
            
        edge_index = permutations(np.arange(node_id-subgraph_node_idx, node_id), 2)
        edge_index = np.array(list(edge_index)).transpose()
        if edge_index.shape[0] == 0:
            edge_index = np.array([[node_id-subgraph_node_idx],[node_id-subgraph_node_idx]])
        edge_index_list.append(edge_index)
        
    agent_trajs = get_past_current_traj(parsed, np.expand_dims(trajectory_ids, axis=1))
    for k in range(agent_trajs.shape[0]):
        agent_pos = agent_trajs[k]
        
        subgraph_node_idx = 0
        for l in range(0, agent_pos.shape[0], time_interval):
            if l == agent_pos.shape[0] - 1:
                break
            else:
                start = agent_pos[l] - origin
                if l > agent_pos.shape[0] - 1 - time_interval:
                    end = agent_pos[agent_pos.shape[0] - 1] - origin
                else:
                    end = agent_pos[l+time_interval] -origin
            a_type = [0]
            node =  np.concatenate([start, end, a_type, [k]])
            node_list.append(node)
            subgraph_node_idx += 1
            node_id += 1
            
        edge_index = permutations(np.arange(node_id-subgraph_node_idx, node_id), 2)
        edge_index = np.array(list(edge_index)).transpose()
        if edge_index.shape[0] == 0:
            edge_index = np.array([[node_id-subgraph_node_idx],[node_id-subgraph_node_idx]])
        edge_index_list.append(edge_index)
    
    nodes = np.stack(node_list, axis=0)
    final_edge_index = np.concatenate(edge_index_list, axis=1)
    
    nodes = torch.tensor(nodes, dtype=torch.float)
    final_edge_index = torch.tensor(final_edge_index, dtype=torch.long)
    
    data = Data(x=nodes, edge_index=final_edge_index)
    
def if_vehicle_interaction(parsed):
    ia_idx = tf.where((parsed['state/objects_of_interest']==1)).numpy()
    if ia_idx.shape[0] == 2:
        if parsed['state/type'][ia_idx[0][0]] == 1 and parsed['state/type'][ia_idx[1][0]] == 1:
            return True
    return False