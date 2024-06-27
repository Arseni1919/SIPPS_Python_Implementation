import re
import os
import math
import json
import time
import heapq
import random
import pstats
import cProfile
import itertools
from itertools import combinations, permutations, tee, pairwise
from typing import *

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class Node:
    def __init__(self, x: int, y: int, neighbours: List[str] | None = None):
        self.x = x
        self.y = y
        self.neighbours = [] if neighbours is None else neighbours
        self.xy_name = f'{self.x}_{self.y}'

    @property
    def xy(self):
        return self.x, self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.xy_name < other.xy_name

    def __gt__(self, other):
        return self.xy_name > other.xy_name

    def __hash__(self):
        return hash(self.xy_name)

    def get_pattern(self) -> dict:
        return {'x': self.x, 'y': self.y, 'neighbours': self.neighbours}


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# HELP FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def set_seed(random_seed_bool, seed=1):
    if random_seed_bool:
        seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    print(f'[SEED]: --- {seed} ---')


def get_dims_from_pic(img_dir: str, path: str = 'maps') -> Tuple[int, int]:
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height = int(re.search(r'\d+', lines[1]).group())
        width = int(re.search(r'\d+', lines[2]).group())
    return height, width


def get_np_from_dot_map(img_dir: str, path: str = 'maps') -> Tuple[np.ndarray, Tuple[int, int]]:
    with open(f'{path}/{img_dir}') as f:
        lines = f.readlines()
        height, width = get_dims_from_pic(img_dir, path)
        img_np = np.zeros((height, width))
        for height_index, line in enumerate(lines[4:]):
            for width_index, curr_str in enumerate(line):
                if curr_str == '.':
                    img_np[height_index, width_index] = 1
        return img_np, (height, width)


def build_graph_from_np(img_np: np.ndarray, show_map: bool = False) -> Tuple[List[Node], Dict[str, Node]]:
    # 0 - wall, 1 - free space
    nodes = []
    nodes_dict = {}

    x_size, y_size = img_np.shape
    # CREATE NODES
    for i_x in range(x_size):
        for i_y in range(y_size):
            if img_np[i_x, i_y] == 1:
                node = Node(i_x, i_y)
                nodes.append(node)
                nodes_dict[node.xy_name] = node

    # CREATE NEIGHBOURS
    for node1, node2 in combinations(nodes, 2):
        if abs(node1.x - node2.x) > 1 or abs(node1.y - node2.y) > 1:
            continue
        if abs(node1.x - node2.x) == 1 and abs(node1.y - node2.y) == 1:
            continue
        node1.neighbours.append(node2.xy_name)
        node2.neighbours.append(node1.xy_name)
        # dist = distance_nodes(node1, node2)
        # if dist == 1:

    for curr_node in nodes:
        curr_node.neighbours.append(curr_node.xy_name)
        heapq.heapify(curr_node.neighbours)

    if show_map:
        plt.imshow(img_np, cmap='gray', origin='lower')
        plt.show()
        # plt.pause(1)
        # plt.close()

    return nodes, nodes_dict


def exctract_h_dict(img_dir, path) -> Dict[str, np.ndarray]:
    # print(f'Started to build heuristic for {kwargs['img_dir'][:-4]}...')
    possible_dir = f"{path}/h_dict_of_{img_dir[:-4]}.json"

    # if there is one
    if os.path.exists(possible_dir):
        # Opening JSON file
        with open(possible_dir, 'r') as openfile:
            # Reading from json file
            h_dict = json.load(openfile)
            for k, v in h_dict.items():
                h_dict[k] = np.array(v)
            return h_dict

    raise RuntimeError('nu nu')


def create_constraints(
        paths: List[List[Node]], map_dim: Tuple[int, int]
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    vc_np: vertex constraints [x, y, t] = bool
    ec_np: edge constraints [x, y, x, y, t] = bool
    pc_np: permanent constraints [x, y] = int or -1
    """
    if len(paths) == 0:
        return None, None, None
    max_path_len = max(map(lambda x: len(x), paths))
    if max_path_len == 0:
        return None, None, None
    vc_np = np.zeros((map_dim[0], map_dim[1], max_path_len))
    ec_np = np.zeros((map_dim[0], map_dim[1], map_dim[0], map_dim[1], max_path_len))
    pc_np = np.ones((map_dim[0], map_dim[1])) * -1
    for path in paths:
        update_constraints(path, vc_np, ec_np, pc_np)
    return vc_np, ec_np, pc_np


def init_constraints(
        map_dim: Tuple[int, int], max_path_len: int
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    vc_np: vertex constraints [x, y, t] = bool
    ec_np: edge constraints [x, y, x, y, t] = bool
    pc_np: permanent constraints [x, y] = int or -1
    """
    if max_path_len == 0:
        return None, None, None
    vc_np = np.zeros((map_dim[0], map_dim[1], max_path_len))
    ec_np = np.zeros((map_dim[0], map_dim[1], map_dim[0], map_dim[1], max_path_len))
    pc_np = np.ones((map_dim[0], map_dim[1])) * -1
    return vc_np, ec_np, pc_np


def update_constraints(
        path: List[Node], vc_np: np.ndarray, ec_np: np.ndarray, pc_np: np.ndarray
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    vc_np: vertex constraints [x, y, t] = bool
    ec_np: edge constraints [x, y, x, y, t] = bool
    pc_np: permanent constraints [x, y] = int or -1
    """
    # pc
    last_node = path[-1]
    last_time = len(path) - 1
    pc_np[last_node.x, last_node.y] = max(pc_np[last_node.x, last_node.y], last_time)
    prev_n = path[0]
    for t, n in enumerate(path):
        # vc
        vc_np[n.x, n.y, t] = 1
        # ec
        ec_np[prev_n.x, prev_n.y, n.x, n.y, t] = 1
        prev_n = n
    return vc_np, ec_np, pc_np


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# SIPPS FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def get_si_table(
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        vc_hard_np: np.ndarray,  # x, y, t -> bool (0/1)
        pc_hard_np: np.ndarray,  # x, y -> time (int)
        vc_soft_np: np.ndarray,  # x, y, t -> bool (0/1)
        pc_soft_np: np.ndarray,  # x, y -> time (int)
):
    """
    safe interval for a vertex is a contiguous period of time during which:
    (1) there are no hard vertex obstacles and no hard target obstacles
    and
    (2) there is either
        (a) a soft vertex or target obstacle at every timestep
        or
        (b) no soft vertex obstacles and no soft target obstacles at any timestep.
    """
    inf_num = 1e10
    si_table: Dict[str, List[Tuple[int, int]]] = {n.xy_name: [] for n in nodes}
    max_t_len = int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1  # index starts at 0

    for n in nodes:

        for i_time in range(max_t_len):



            print(f'\r{n.xy_name} - {i_time}', end='')



# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

