import heapq

import numpy as np

from sipps_main_funcitons import *


def run_sipps_insert_node():
    pass


def run_sipps_expand_node():
    pass


def run_sipps(
        curr_node: Node,
        goal_node: Node,
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        vc_hard_np: np.ndarray,  # x, y, t -> bool (0/1)
        ec_hard_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
        pc_hard_np: np.ndarray,  # x, y -> time (int)
        vc_soft_np: np.ndarray,  # x, y, t -> bool (0/1)
        ec_soft_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
        pc_soft_np: np.ndarray,  # x, y -> time (int)
        inf_num: int = int(1e10)
) -> List[Node] | None:

    assert vc_hard_np.shape[2] == vc_soft_np.shape[2]
    assert ec_hard_np.shape[4] == ec_soft_np.shape[4]
    assert int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1 == vc_hard_np.shape[2]
    assert int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1 == ec_hard_np.shape[4]
    assert pc_hard_np[goal_node.x, goal_node.y] != 1

    si_table = get_si_table(nodes, nodes_dict, vc_hard_np, pc_hard_np, vc_soft_np, pc_soft_np, inf_num)
    root = SIPPSNode(curr_node, si_table[curr_node.xy_name][0], 0, False)
    T = 0
    goal_vc_times_list = get_o_h(goal_node, vc_hard_np)
    if len(goal_vc_times_list) > 0:
        T = max(goal_vc_times_list) + 1
    goal_vc_times_list = get_o_h(goal_node, vc_soft_np)
    T_tag = T
    if len(goal_vc_times_list) > 0:
        T_tag = max(T, max(goal_vc_times_list) + 1)
    goal_np: np.ndarray = h_dict[goal_node.xy_name]
    compute_c_g_h_f_values(root, goal_node, goal_np, T, T_tag, vc_soft_np, ec_soft_np, pc_soft_np)

    Q: List[SIPPSNode] = []
    P: List[SIPPSNode] = []
    heapq.heappush(Q, root)

    while len(Q) > 0:
        next_n: SIPPSNode = heapq.heappop(Q)
        if next_n.is_goal:
            return extract_path(next_n)
        if next_n.n == goal_node and next_n.low >= T:
            pass


def main():
    # set_seed(random_seed_bool=False, seed=7310)
    # set_seed(random_seed_bool=False, seed=123)
    set_seed(random_seed_bool=True)

    # img_dir = '10_10_my_rand.map'
    img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'maze-32-32-4.map'

    to_render: bool = True
    # to_render: bool = False

    path_to_maps: str = 'maps'
    path_to_heuristics: str = 'logs_for_heuristics'

    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    h_dict = exctract_h_dict(img_dir, path_to_heuristics)

    path1 = [
        nodes_dict['7_1'],
        nodes_dict['7_1'],
        nodes_dict['6_1'],
        nodes_dict['5_1'],
        nodes_dict['4_1'],
    ]
    path2 = [
        nodes_dict['6_0'],
        nodes_dict['6_0'],
        nodes_dict['6_0'],
        nodes_dict['6_0'],
        nodes_dict['6_1'],
        nodes_dict['6_1'],
        nodes_dict['6_1'],
        nodes_dict['6_0'],
    ]
    paths = [path1, path2]
    max_path_len = max(map(lambda x: len(x), paths))
    vc_hard_np, ec_hard_np, pc_hard_np = create_constraints(paths, map_dim, max_path_len)

    path = [
        nodes_dict['7_2'],
        nodes_dict['7_2'],
        nodes_dict['6_2'],
        nodes_dict['5_2'],
        nodes_dict['4_2'],
    ]
    paths = [path]
    vc_soft_np, ec_soft_np, pc_soft_np = create_constraints(paths, map_dim, max_path_len)

    start_node = nodes_dict['6_2']
    goal_node = nodes_dict['6_1']

    run_sipps(
        start_node, goal_node, nodes, nodes_dict, h_dict,
        vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np
    )

    print()


if __name__ == '__main__':
    main()
