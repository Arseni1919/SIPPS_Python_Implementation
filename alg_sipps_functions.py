from functions import *


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# CLASSES
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


class SIPPSNode:
    def __init__(self, n: Node, si: Tuple[int, int], _id: int, is_goal: bool, parent: Self | None = None):
        self.x: int = n.x
        self.y: int = n.y
        self.n = n
        self.neighbours = n.neighbours
        # random.shuffle(self.neighbours)
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
# SIPPS FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def get_si_table(
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        vc_hard_np: np.ndarray | None,  # x, y, t -> bool (0/1)
        pc_hard_np: np.ndarray | None,  # x, y -> time (int)
        vc_soft_np: np.ndarray | None,  # x, y, t -> bool (0/1)
        pc_soft_np: np.ndarray | None,  # x, y -> time (int)
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


def get_vc_list(
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
    if pc_soft_np[sipps_node.x, sipps_node.y] >= sipps_node.low:
        return 1
    vc_si_list = vc_soft_np[sipps_node.x, sipps_node.y, sipps_node.low: sipps_node.high]
    if np.sum(vc_si_list) > 0:
        return 1
    return 0


def get_c_e(
        sipps_node: SIPPSNode,
        ec_soft_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
) -> int:
    if sipps_node.parent is None:
        return 0
    parent = sipps_node.parent
    if sipps_node.low < ec_soft_np.shape[4] and ec_soft_np[sipps_node.x, sipps_node.y, parent.x, parent.y, sipps_node.low] == 1:
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
        c_e = get_c_e(sipps_node, ec_soft_np)
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


def extract_path(next_sipps_node: SIPPSNode) -> Tuple[List[Node], Deque[SIPPSNode]]:
    sipps_path: Deque[SIPPSNode] = deque([next_sipps_node])
    sipps_path_save: Deque[SIPPSNode] = deque([next_sipps_node])
    parent = next_sipps_node.parent
    while parent is not None:
        sipps_path.appendleft(parent)
        sipps_path_save.appendleft(parent)
        parent = parent.parent

    path_with_waiting: List[Node] = []
    while len(sipps_path) > 0:
        next_node = sipps_path.popleft()
        path_with_waiting.append(next_node.n)
        if len(sipps_path) == 0:
            break
        if len(path_with_waiting) - 1 < sipps_path[0].low:
            path_with_waiting.append(path_with_waiting[-1])
    return path_with_waiting, sipps_path_save


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


def get_I_group(
        node: SIPPSNode,
        nodes_dict: Dict[str, Node],
        si_table: Dict[str, List[Tuple[int, int]]],
) -> List[Tuple[Node, int]]:
    I_group: List[Tuple[Node, int]] = []
    for nei_name in node.neighbours:
        nei_si_list = si_table[nei_name]
        if nei_name == node.xy_name:
            for si_id, si in enumerate(nei_si_list):
                if si[0] == node.high:
                    I_group.append((node.n, si_id))  # indicates wait action
                    break
            continue
        for si_id, si in enumerate(nei_si_list):
            if ranges_intersect(range1=(si[0], si[1] - 1), range2=(node.low + 1, node.high)):
                I_group.append((nodes_dict[nei_name], si_id))
                continue
    return I_group


def get_low_without_hard_ec(
        from_node: Node,
        to_node: Node,
        init_low: int,
        init_high: int,
        ec_hard_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
) -> int | None:
    for i_t in range(init_low, init_high):
        if i_t >= ec_hard_np.shape[4]:
            return 0
        if ec_hard_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0:
            return i_t
    return None


def get_low_without_hard_and_soft_ec(
        from_node: Node,
        to_node: Node,
        new_low: int,
        init_high: int,
        ec_hard_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
        ec_soft_np: np.ndarray,  # x, y, x, y, t -> bool (0/1)
) -> int | None:
    for i_t in range(new_low, init_high):
        if i_t >= ec_hard_np.shape[4]:
            return 0
        no_in_h = ec_hard_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0
        no_in_s = ec_soft_np[to_node.x, to_node.y, from_node.x, from_node.y, i_t] == 0
        if no_in_h and no_in_s:
            return i_t
    return None


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

