from alg_LNS2_functions import *


def run_lns2(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        # constr_type: str = 'hard',
        constr_type: str = 'soft',
        n_neighbourhood: int = 5,
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
    the number of collisions with each other and with the paths in P \\ P−.
    Speciﬁcally, MAPF-LNS2 uses a modiﬁcation of Prioritized Planning (PP) as the modiﬁed MAPF algorithm.
    PP assigns a random priority ordering to the agents in As and replans their paths one at a time according
    to the ordering. Each time, it calls a single-agent pathﬁnding algorithm (see Section 4) to ﬁnd a path for
    an agent that minimizes the number of collisions with the new paths of the higher-priority agents in As and
    the paths in P \\ P−. We denote the new paths of the agents in As as P+.
    - Finally, MAPF-LNS2 replaces the old plan P with the new plan (P \\ P−) ∪ P+ iff the number of colliding pairs
    (CP) of the paths in the new plan is no larger than that of the old plan.
    """
    start_time = time.time()
    # create agents
    agents: List[AgentLNS2] = []
    agents_dict: Dict[str, AgentLNS2] = {}
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentLNS2(num, s_node, g_node)
        agents.append(new_agent)
        agents_dict[new_agent.name] = new_agent

    # init solution
    create_init_solution(agents, nodes, nodes_dict, h_dict, map_dim, constr_type, start_time)
    cp_graph, cp_graph_names = get_cp_graph(agents)
    cp_len = len(cp_graph)

    # repairing procedure
    while cp_len > 0:
        print(f'\n{cp_len=}')
        agents_subset: List[AgentLNS2] = get_agents_subset(cp_graph, cp_graph_names, n_neighbourhood, agents)
        old_paths: Dict[str, List[Node]] = {a.name: a.path[:] for a in agents_subset}
        agents_outer: List[AgentLNS2] = [a for a in agents if a not in agents_subset]

        # assert len(set(agents_outer)) == len(agents_outer)
        # assert len(set(agents_subset)) == len(agents_subset)
        # assert len(set(agents)) == len(agents)
        # assert len(agents_subset) + len(agents_outer) == len(agents)

        solve_subset_with_prp(agents_subset, agents_outer, nodes, nodes_dict, h_dict, map_dim, start_time, constr_type, agents)

        cp_graph, cp_graph_names = get_cp_graph(agents)
        if len(cp_graph) > cp_len:
            for agent in agents_subset:
                agent.path = old_paths[agent.name]
            cp_graph, cp_graph_names = get_cp_graph(agents)
            continue
        cp_len = len(cp_graph)

    return {a.name: a.path for a in agents}, {'agents': agents}


@use_profiler(save_dir='stats/alg_lns2.pstat')
def main():
    # set_seed(random_seed_bool=False, seed=7310)
    set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)

    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    # img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'maze-32-32-4.map'

    n_agents = 200

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
