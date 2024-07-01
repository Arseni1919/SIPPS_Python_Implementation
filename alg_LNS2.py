from alg_LNS2_functions import *
from alg_sipps import run_sipps


def create_init_solution(
        agents: List[AgentLNS2],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        constr_type: str,
        start_time: int | float
):
    c_sum: int = 0
    h_priority_agents: List[AgentLNS2] = []
    for agent in agents:
        (vc_hard_np, ec_hard_np, pc_hard_np,
         vc_soft_np, ec_soft_np, pc_soft_np) = create_hard_and_soft_constraints(h_priority_agents, map_dim,
                                                                                constr_type)
        new_path, sipps_info = run_sipps(
            agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
            vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, agent=agent
        )
        if new_path is None:
            agent.path = None
            break
        agent.path = new_path[:]
        h_priority_agents.append(agent)
        c_sum += sipps_info['c']

        # checks
        runtime = time.time() - start_time
        print(f'\r[init] | agents: {len(h_priority_agents): <3} / {len(agents)} | {runtime= : .2f} s.')  # , end=''
        collisions: int = 0
        align_all_paths(h_priority_agents)
        for i in range(len(h_priority_agents[0].path)):
            to_count = False if constr_type == 'hard' else True
            collisions += check_vc_ec_neic_iter(h_priority_agents, i, to_count)
        if collisions > 0:
            print(f'{collisions=} | {c_sum=}')


def run_lns2(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        # constr_type: str = 'hard',
        constr_type: str = 'soft',
        time_limit: int = 60  # seconds
) -> Tuple[Dict[str, List[Node]] | None, dict]:
    """
    To begin with,
    - MAPF-LNS2 calls a MAPF algorithm to solve the instance and obtains a (partial or complete) plan
    from the MAPF algorithm.
    - For each agent that does not yet have a path, MAPF-LNS2 plans a path for it that
    minimizes the number of collisions with the existing paths (Section 4).
    - MAPF-LNS2 then repeats a repairing procedure until the plan P becomes
    feasible.
    - At each iteration, MAPF-LNS2 selects a subset of agents As ⊆ A by a neighborhood selection method (see
    Section 5). We denote the paths of the agents in As as P−.
    - It then calls a modiﬁed MAPF algorithm to replan the paths of the agents in As to minimize
    the number of collisions with each other and with the paths in P\ P−.
    Speciﬁcally, MAPF-LNS2 uses a modiﬁcation of Prioritized Planning (PP) as the modiﬁed MAPF algorithm.
    PP assigns a random priority ordering to the agents in As and replans their paths one at a time according
    to the ordering. Each time, it calls a single-agent pathﬁnding algorithm (see Section 4) to ﬁnd a path for
    an agent that minimizes the number of collisions with the new paths of the higher-priority agents in As and
    the paths in P \ P−. We denote the new paths of the agents in As as P+.
    - Finally, MAPF-LNS2 replaces the old plan P with the new plan (P \ P−) ∪ P+ iff the number of colliding pairs
    (CP) of the paths in the new plan is no larger than that of the old plan.
    """
    start_time = time.time()
    # create agents
    agents = []
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentLNS2(num, s_node, g_node)
        agents.append(new_agent)

    # init solution
    create_init_solution(agents, nodes, nodes_dict, h_dict, map_dim, constr_type, start_time)

    # repairing procedure
    pass

    return {a.name: a.path for a in agents}, {'agents': agents}


@use_profiler(save_dir='stats/alg_lns2.pstat')
def main():
    # set_seed(random_seed_bool=False, seed=7310)
    set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)

    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    img_dir = 'maze-32-32-4.map'

    n_agents = 50

    to_render: bool = True
    # to_render: bool = False

    path_to_maps: str = 'maps'
    path_to_heuristics: str = 'logs_for_heuristics'

    img_np, (height, width) = get_np_from_dot_map(img_dir, path_to_maps)
    map_dim = (height, width)
    nodes, nodes_dict = build_graph_from_np(img_np, show_map=False)
    h_dict = exctract_h_dict(img_dir, path_to_heuristics)

    start_nodes: List[Node] = random.sample(nodes, n_agents)
    goal_nodes: List[Node] = random.sample(nodes, n_agents)

    paths_dict, info = run_lns2(
        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim
    )

    # plot
    if to_render and paths_dict is not None:
        agents: List[AgentLNS2] = info['agents']
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        plot_rate = 0.001
        # plot_rate = 0.5
        # plot_rate = 1
        max_path_len = max([len(a.path) for a in agents])

        for i in range(max_path_len):
            # update curr nodes
            for a in agents:
                a.update_curr_node(i)
            # plot the iteration
            i_agent = agents[0]
            plot_info = {
                'img_np': img_np,
                'agents': agents,
                'i_agent': i_agent,
            }
            plot_step_in_env(ax[0], plot_info)
            plt.pause(plot_rate)
        plt.show()


if __name__ == '__main__':
    main()
