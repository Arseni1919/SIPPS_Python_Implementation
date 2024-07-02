import random
import time

import matplotlib.pyplot as plt

from alg_PrP_functions import *
from alg_sipps import run_sipps


def run_prp(
        start_nodes: List[Node],
        goal_nodes: List[Node],
        nodes: List[Node],
        nodes_dict: Dict[str, Node],
        h_dict: Dict[str, np.ndarray],
        map_dim: Tuple[int, int],
        constr_type: str = 'hard',
        # constr_type: str = 'soft',
        time_limit: int = 60  # seconds
) -> Tuple[Dict[str, List[Node]] | None, dict]:

    start_time = time.time()

    # create agents
    agents = []
    for num, (s_node, g_node) in enumerate(zip(start_nodes, goal_nodes)):
        new_agent = AgentPrP(num, s_node, g_node)
        agents.append(new_agent)

    r_iter = 0
    while time.time() - start_time < time_limit:
        # calc paths
        h_priority_agents: List[AgentPrP] = []
        for agent in agents:
            (vc_hard_np, ec_hard_np, pc_hard_np,
             vc_soft_np, ec_soft_np, pc_soft_np) = create_hard_and_soft_constraints(h_priority_agents, map_dim, constr_type)
            new_path, sipps_info = run_sipps(
                agent.start_node, agent.goal_node, nodes, nodes_dict, h_dict,
                vc_hard_np, ec_hard_np, pc_hard_np, vc_soft_np, ec_soft_np, pc_soft_np, agent=agent
            )
            if new_path is None:
                agent.path = None
                break
            agent.path = new_path[:]
            h_priority_agents.append(agent)

            # checks
            runtime = time.time() - start_time
            print(f'\r{r_iter=: <3} | agents: {len(h_priority_agents): <3} / {len(agents)} | {runtime= : .2f} s.')  # , end=''
            collisions: int = 0
            align_all_paths(h_priority_agents)
            for i in range(len(h_priority_agents[0].path)):
                to_count = False if constr_type == 'hard' else True
                collisions += check_vc_ec_neic_iter(h_priority_agents, i, to_count)
            if collisions > 0:
                print(f'{collisions=} | {sipps_info['c']=}')

        # return check
        to_return = True
        for agent in agents:
            if agent.path is None:
                to_return = False
                break
            if agent.path[-1] != agent.goal_node:
                to_return = False
                break
        if to_return:
            return {a.name: a.path for a in agents}, {'agents': agents}

        # reshuffle
        r_iter += 1
        random.shuffle(agents)
        for agent in agents:
            agent.path = []

    return None, {}


@use_profiler(save_dir='stats/alg_prp.pstat')
def main():
    # set_seed(random_seed_bool=False, seed=7310)
    set_seed(random_seed_bool=False, seed=123)
    # set_seed(random_seed_bool=True)

    # img_dir = '10_10_my_rand.map'
    # img_dir = 'empty-32-32.map'
    # img_dir = 'random-32-32-10.map'
    # img_dir = 'random-32-32-20.map'
    img_dir = 'room-32-32-4.map'
    # img_dir = 'maze-32-32-2.map'
    # img_dir = 'maze-32-32-4.map'

    n_agents = 100

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

    paths_dict, info = run_prp(
        start_nodes, goal_nodes, nodes, nodes_dict, h_dict, map_dim
    )

    # plot
    if to_render and paths_dict is not None:
        agents: List[AgentPrP] = info['agents']
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







