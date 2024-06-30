from typing import *
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


color_names = [
    # 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # Single-letter abbreviations
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',  # Full names
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


def get_color(i):
    index_to_pick = i % len(color_names)
    return color_names[index_to_pick]


def do_the_animation(info, to_save=False, to_animate=False):
    if not to_save and not to_animate:
        return
    img_np: np.ndarray = info['img_np']
    paths_dict: Dict[str, List[Any]] = info['paths_dict']
    max_time: int | float = info['max_time']
    img_dir: str = info['img_dir']
    alg_name: str = info['alg_name']
    n_agents: int = len(paths_dict)
    i_agent = info['i_agent']

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    field = img_np * -1
    ax[0].imshow(field, origin='lower')
    ax[0].set_title(f'{n_agents} agents, {max_time} steps')

    goal_scat1 = ax[0].scatter([i_agent.goal_node.y], [i_agent.goal_node.x], s=400, c='white', marker='X', alpha=0.4)
    goal_scat2 = ax[0].scatter([i_agent.goal_node.y], [i_agent.goal_node.x], s=200, c='red', marker='X', alpha=0.4)

    others_y_list, others_x_list, others_cm_list = [], [], []
    for i, (agent_name, path) in enumerate(paths_dict.items()):
        others_y_list.append(path[0].y)
        others_x_list.append(path[0].x)
        others_cm_list.append(get_color(i))
    scat1 = ax[0].scatter(others_y_list, others_x_list, s=100, c='k')
    scat2 = ax[0].scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))

    agent_scat1 = ax[0].scatter([i_agent.start_node.y], [i_agent.start_node.x], s=120, c='w')
    agent_scat2 = ax[0].scatter([i_agent.start_node.y], [i_agent.start_node.x], s=70, c='r')

    def update(frame):

        fr_i_goal = i_agent.goal_node
        data = np.stack([[fr_i_goal.y], [fr_i_goal.x]]).T
        goal_scat1.set_offsets(data)
        goal_scat2.set_offsets(data)

        # for each frame, update the data stored on each artist.
        fr_y_list, fr_x_list = [], []
        for i, (agent_name, path) in enumerate(paths_dict.items()):
            fr_node = path[frame]
            fr_y_list.append(fr_node.y)
            fr_x_list.append(fr_node.x)
        # update the scatter plot:
        data = np.stack([fr_y_list, fr_x_list]).T
        scat1.set_offsets(data)
        scat2.set_offsets(data)

        fr_i_node = i_agent.path[frame]
        data = np.stack([[fr_i_node.y], [fr_i_node.x]]).T
        agent_scat1.set_offsets(data)
        agent_scat2.set_offsets(data)

        return goal_scat1, goal_scat2, scat1, scat2, agent_scat1, agent_scat2

    ani = animation.FuncAnimation(fig=fig, func=update, frames=max_time, interval=250)
    if to_save:
        add_text = f'{alg_name}_'
        ani.save(filename=f"../videos/{add_text}{n_agents}_agents_in_{img_dir[:-4]}_for_{max_time}_steps.mp4", writer="ffmpeg")
    if to_animate:
        plt.show()

