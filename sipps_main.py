import numpy as np

from sipps_main_funcitons import *


class SIPPSNode:
    def __init__(self, x: int, y: int, si: Tuple[int, int], _id: int, is_goal: bool):
        self.x: int = x
        self.y: int = y
        self.xy_name: str = f'{x}_{y}'
        self.si: Tuple[int, int] = si
        self._id: int = _id
        self.is_goal: bool = is_goal

        self.g: int = 0
        self.t: int = 0
        self.c: int = 0

    @property
    def f(self):
        return self.g + self.t


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
):

    assert vc_hard_np.shape[2] == vc_soft_np.shape[2]
    assert ec_hard_np.shape[4] == ec_soft_np.shape[4]
    assert int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1 == vc_hard_np.shape[2]
    assert int(max(np.max(pc_hard_np), np.max(pc_soft_np))) + 1 == ec_hard_np.shape[4]

    si_table = get_si_table(nodes, nodes_dict, vc_hard_np, pc_hard_np, vc_soft_np, pc_soft_np)



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

    path = [
        nodes_dict['7_1'],
        nodes_dict['7_1'],
        nodes_dict['6_1'],
        nodes_dict['5_1'],
        nodes_dict['4_1'],
    ]
    paths = [path]
    vc_hard_np, ec_hard_np, pc_hard_np = create_constraints(paths, map_dim)

    path = [
        nodes_dict['7_2'],
        nodes_dict['7_2'],
        nodes_dict['6_2'],
        nodes_dict['5_2'],
        nodes_dict['4_2'],
    ]
    paths = [path]
    vc_soft_np, ec_soft_np, pc_soft_np = create_constraints(paths, map_dim)

    start_node = nodes_dict['6_1']
    goal_node = nodes_dict['6_2']

    run_sipps(
        start_node, goal_node, nodes, nodes_dict, h_dict,
        vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np
    )

    print()


if __name__ == '__main__':
    main()
