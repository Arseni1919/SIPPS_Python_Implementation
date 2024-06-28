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


class SIPPSNode:
    def __init__(self, n: Node, si: Tuple[int, int], _id: int, is_goal: bool, parent: Self | None = None):
        self.x: int = n.x
        self.y: int = n.y
        self.n = n
        # self.xy_name: str = f'{self.x}_{self.y}'
        self.xy_name: str = self.n.xy_name
        self.si: List[int] = [si[0], si[1]]
        self._id: int = _id
        self.is_goal: bool = is_goal
        self.parent: Self = parent

        self.g: int = 0
        self.h: int = 0
        self.f: int = 0
        self.c: int = 0

    @property
    def low(self):
        return self.si[0]

    @property
    def high(self):
        return self.si[1]

    @property
    def id(self):
        return self._id

    def set_low(self, new_v: int):
        self.si[0] = new_v

    def set_high(self, new_v: int):
        self.si[1] = new_v

    def __lt__(self, other: Self):
        if self.c < other.c:
            return True
        if self.c > other.c:
            return False
        if self.f < other.f:
            return True
        if self.f > other.f:
            return False
        if self.h < other.h:
            return True
        if self.h > other.h:
            return False
        if self.xy_name < other.xy_name:
            return True
        return False


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
        paths: List[List[Node]], map_dim: Tuple[int, int], max_path_len: int
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    vc_np: vertex constraints [x, y, t] = bool
    ec_np: edge constraints [x, y, x, y, t] = bool
    pc_np: permanent constraints [x, y] = int or -1
    """
    if len(paths) == 0:
        return None, None, None
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
        inf_num: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    safe interval for a vertex is a contiguous period of time during which:
    (1) there are no hard vertex obstacles and no hard target obstacles
    and
    (2) there is either
        (a) a soft vertex or target obstacle at every timestep
        or
        (b) no soft vertex obstacles and no soft target obstacles at any timestep.
    """
    si_table: Dict[str, List[Tuple[int, int]]] = {n.xy_name: [] for n in nodes}
    max_t_len = int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1  # index starts at 0

    for n in nodes:

        curr_pc_hard = pc_hard_np[n.x, n.y] if pc_hard_np[n.x, n.y] != -1 else inf_num
        curr_pc_soft = pc_soft_np[n.x, n.y] if pc_soft_np[n.x, n.y] != -1 else inf_num
        v_line = []
        after_pc_hard = False
        after_pc_soft = False
        for i_time in range(max_t_len + 1):
            # check pc
            if not after_pc_hard and i_time >= curr_pc_hard:
                after_pc_hard = True
            if not after_pc_soft and i_time >= curr_pc_soft:
                after_pc_soft = True
            # check
            if after_pc_hard:
                v_line.append(1)
                continue
            if i_time < max_t_len and vc_hard_np[n.x, n.y, i_time] == 1:
                v_line.append(1)
                continue
            if after_pc_soft:
                v_line.append(0.5)
                continue
            if i_time < max_t_len and vc_soft_np[n.x, n.y, i_time] == 1:
                v_line.append(0.5)
                continue
            v_line.append(0)
        v_line.append(inf_num)

        start_si_time = 0
        started_si = False
        si_type = 0
        for i_time, i_value in enumerate(v_line):
            if i_value == inf_num:
                assert i_time == len(v_line) - 1
                if v_line[i_time-1] == 1:
                    break
                # CLOSE
                si_table[n.xy_name].append((start_si_time, inf_num))
                break
            if i_value == 1:
                if started_si:
                    # CLOSE
                    si_table[n.xy_name].append((start_si_time, i_time))
                started_si = False
                continue
            if not started_si:
                started_si = True
                start_si_time = i_time
                si_type = i_value
                continue
            # if you here -> the i is 0.5 / 0 / inf
            if si_type != i_value:
                # CLOSE
                si_table[n.xy_name].append((start_si_time, i_time))
                start_si_time = i_time
                si_type = i_value

        # print(f'{n.xy_name}: {v_line} -> {si_table[n.xy_name]}')
    return si_table


def get_o_h(
        node: Node,
        vc_np: np.ndarray,  # x, y, t -> bool (0/1)
) -> List[int]:
    filter_arr = np.argwhere(vc_np[node.x, node.y, :] > 0)
    vc_list = [int(x[0]) for x in filter_arr]
    return vc_list


def get_c_v(
        sipps_node: SIPPSNode,
        vc_soft_np: np.ndarray,  # x, y, t -> bool (0/1)
        pc_soft_np: np.ndarray,  # x, y -> time (int)
) -> int:
    if pc_soft_np[sipps_node.x, sipps_node.y] > sipps_node.low:
        return 1
    vc_si_list = vc_soft_np[sipps_node.x, sipps_node.y, sipps_node.low:sipps_node.high]
    if np.any(vc_si_list):
        return 1
    return 0

def get_c_e(
        sipps_node: SIPPSNode,
        ec_soft_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
) -> int:
    if sipps_node.parent is None:
        return 0
    parent = sipps_node.parent
    if ec_soft_np[sipps_node.x, sipps_node.y, parent.x, parent.y, sipps_node.low] == 1:
        return 1
    return 0


def compute_c_g_h_f_values(
        sipps_node: SIPPSNode,
        goal_node: Node,
        goal_np: np.ndarray,
        T: int,
        T_tag: int,
        vc_soft_np: np.ndarray,  # x, y, t -> bool (0/1)
        ec_soft_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
        pc_soft_np: np.ndarray,  # x, y -> time (int)
) -> None:
    # c
    """
    Each node n also maintains a c-value, which is
    the (underestimated) number of the soft collisions of the partial path from the root node to node n, i.e.,
    c(n) = c(n`) + cv + ce,
    where n` is the parent node of n,
    cv is 1 if the safe interval of n contains soft vertex/target obstacles and 0 otherwise,
    and ce is 1 if ((n`.v, n.v), n.low) ∈ Os and 0 otherwise.
    If n is the root node (i.e., n` does not exist), c(n) = cv.
    """
    c_v = get_c_v(sipps_node, vc_soft_np, pc_soft_np)
    if sipps_node.parent is None:
        sipps_node.c = c_v
    else:
        c_e =get_c_e(sipps_node, ec_soft_np)
        sipps_node.c = sipps_node.parent.c + c_v + c_e

    # g
    if sipps_node.parent is None:
        sipps_node.g = 0
    else:
        sipps_node.g = sipps_node.parent.g + 1

    # h
    if sipps_node.xy_name != goal_node.xy_name:
        d_n = goal_np[sipps_node.x, sipps_node.y]
        if sipps_node.c == 0:
            sipps_node.h = max(d_n, T_tag - sipps_node.g)
        else:
            sipps_node.h = max(d_n, T - sipps_node.g)

    else:
        sipps_node.h = 0

    # f
    sipps_node.f = sipps_node.g + sipps_node.h


def extract_path(next_sipps_node: SIPPSNode) -> List[Node]:
    path = [next_sipps_node.n]
    parent = next_sipps_node.parent
    while parent is not None:
        path.append(parent.n)
        parent = parent.parent
    path.reverse()
    return path


def get_c_future(
        goal_node: Node,
        t: int,
        vc_soft_np: np.ndarray,
        pc_soft_np: np.ndarray
) -> int:
    out_value = 0
    pc_value = pc_soft_np[goal_node.x, goal_node.y]
    if pc_value != -1:
        out_value += 1
    vc_values = vc_soft_np[goal_node.x, goal_node.y, t:]
    out_value += np.sum(vc_values)
    return out_value


def duplicate_sipps_node(node: SIPPSNode) -> SIPPSNode:
    """
    def __init__(self, n: Node, si: Tuple[int, int], _id: int, is_goal: bool, parent: Self | None = None):
    self.x: int = n.x
    self.y: int = n.y
    self.n = n
    self.xy_name: str = self.n.xy_name
    self.si: Tuple[int, int] = si
    self._id: int = _id
    self.is_goal: bool = is_goal
    self.parent: Self = parent

    self.g: int = 0
    self.h: int = 0
    self.f: int = 0
    self.c: int = 0
    """
    return_node = SIPPSNode(
        node.n,
        (node.si[0], node.si[1]),
        node.id,
        node.is_goal,
        node.parent
    )
    return_node.g = node.g
    return_node.h = node.h
    return_node.f = node.f
    return_node.c = node.c

    return return_node


def get_identical_nodes(
        node: SIPPSNode,
        Q: List[SIPPSNode],
        P: List[SIPPSNode],
) -> List[SIPPSNode]:
    """
    Two nodes n1 and n2 have the same identity, denoted as n1 ∼ n2, iff:
    (1) n1.v = n2.v
    (2) n1.id = n2.id
    (3) n1.is_goal = n2.is_goal
    """
    identical_nodes: List[SIPPSNode] = []
    curr_xy_name = node.xy_name
    curr_id = node.id
    curr_is_goal = node.is_goal
    for n in [*Q, *P]:
        if n.xy_name == curr_xy_name and n.id == curr_id and n.is_goal == curr_is_goal:
            identical_nodes.append(n)
    return identical_nodes

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

