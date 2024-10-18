import os
import pickle
import argparse
from shutil import copy
from operator import itemgetter
from itertools import count
import logging
import time

import imageio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np

from agents import Drone
from metric import Metric
from multidrone import MultiDroneEnv


class Progress:

    def __init__(self, initial):
        self.initial_time = initial

    def show(self, time_now, users, n_sim):
        print(f'End Run {n_sim:2d} -- Time:{(time_now - self.initial_time):.2f} '
              f's -- Users Connected {users}')


def show_iter(values_iter, n_episode, val_i):
    """
    The number of iterations are show for each episode

    Args:
        values_iter: list of iterations
        n_episode: number of episodes in simulation
        val_i: number of the independent run
    """
    _, ax = plt.subplots()
    ax.bar(np.arange(n_episode), values_iter)
    ax.set_xticks(list(np.arange(0, n_episode, 5)))
    ax.set_xlabel(f'Episodes')
    ax.set_ylabel(f'Num of iterations')
    ax.set_title(f'Run_{val_i}')
    plt.savefig(f'Iter_x_Episode_{val_i}.png', dpi=100)
    plt.close()


def fig_status(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_status_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_status.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_height(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_height_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_height.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_actions(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_actions_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_actions.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_6(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_6.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_efficiency(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_efficiency_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_efficiency.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_11(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_backhaul_drone{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_11.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_12(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_backhaul_global{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_12.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_time(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_time_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_time.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_battery(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_battery_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_battery.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_power(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_power_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_power.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def fig_energy(total_run):
    global_reward = []
    for i in range(total_run):
        a = np.load(f'Run_energy_{i}.npz')
        global_reward.append(a['data'])

    global_reward = np.stack(global_reward)
    with open('fig_energy.pickle', 'wb') as f:
        pickle.dump([global_reward], f)


def function_simulation(run_i=0, n_episodes=5, ep_greedy=0, n_agents=16, frequency="1e09", n_users=200,
                        weight=1, s_render=0, distribution='cluster', step_z=2):
    """
    Simulation drone environment using Q-Learning
    """
    progress = Progress(time.time())
    frequency_list = [float(item) for item in frequency.split(',')]
    if step_z == 1:
        agents = [Drone(frequency_list, 1) for _ in range(n_agents)]
    elif step_z == 2:
        agents = [Drone(frequency_list, 2) for _ in range(n_agents)]

    for index, agent in enumerate(agents):
        agent.name = f'Drone_{index}'

    if ep_greedy == 0:  # e-greedy decay
        epsilon = 1
    else:  # e-greedy fixed value
        epsilon = ep_greedy

    env = MultiDroneEnv(agents, frequency=frequency_list, n_users=n_users, weight=weight, n_run=run_i)
    env.info = distribution
    actions_name = []
    for action_name in agents[0].actions:
        actions_name.append(action_name.name)

    metric = Metric(run_i, n_episodes, actions_name)
    old_obs = env.reset()
    env.dir_sim = f'Run_{run_i}'

    env.epsilon = epsilon

    env.render(filename=f'Episode_0.png')

    l_rate = 0.9
    discount = 0.9

    num_iter_per_episode = 100
    num_max_iter_same_rew = 20

    # Model energy
    energy_iter_episode = 0
    power_iter_episode = 0
    time_tx_iter_episode = 0
    env.val_velocity = 10  # 10 m/s
    best_scenario = [[0, 'best'] for _ in env.agents]
    equal_rew = 0
    best_rew = 0
    iter_x_episode = []
    iteration = 0
    
    for drone in env.agents:
        drone.save_best()
        
    for episode in range(n_episodes):
        if step_z == 1 and episode >= 50:
            for agent in env.agents:
                agent.step_amplitude_z = 1
        efficiency = []
        for id_drone, drone in enumerate(env.agents):
            for iteration in count():

                if not ep_greedy:
                    env.epsilon = np.exp(-iteration / 5)

                # Choice action
                actions_array = [uavs.actions.stop for uavs in env.agents]  # Action selected
                actions_val_array = actions_array.copy()  # TODO: Action validated
                action_ok, action_selected = drone.choice_action(old_obs[id_drone], env.epsilon)
                actions_array[id_drone] = action_selected
                actions_val_array[id_drone] = action_ok

                reward, new_obs, done, _ = env.step(actions_val_array)

                # Learn agents
                drone.learn(old_obs[id_drone], new_obs[id_drone],
                            [l_rate, discount, reward[id_drone], actions_array[id_drone]])

                # Select the best scenario
                actual_scenario = [reward[id_drone], 'actual']
                both_scenario = [best_scenario[id_drone], actual_scenario]
                s_f = sorted(both_scenario, key=itemgetter(0), reverse=True)
                # Update Criteria
                if s_f[0][1] == 'actual':
                    #best_scenario[id_drone].clear()
                    best_scenario[id_drone] = actual_scenario.copy()
                    best_scenario[id_drone][1] = 'best'
                    drone.save_best()
                    for user in env.user_list:
                        user.save_best()

                if best_rew < reward[id_drone]:
                    best_rew = reward[id_drone]
                    equal_rew = 0
                else:
                    equal_rew += 1

                # Update observation spaces
                old_obs = new_obs.copy()

                # Calculate Energy consumption
                # energy, power, time_tx = env.model_energy(velocity)
                energy_iter_episode += env.model_dict.energy
                power_iter_episode += env.model_dict.power
                time_tx_iter_episode += env.model_dict.time_tx
                efficiency.append(env.model_dict.efficiency)
                # Stopping Criteria
                # First Condition
                if iteration == num_iter_per_episode - 1:
                    break

                # Second Condition
                if equal_rew == num_max_iter_same_rew - 1:
                    break

                # New Condition
                if done:
                    break

        # All Iterations End
        equal_rew = 0
        iter_x_episode.append(iteration)
        # Load best scenario
        # save_pos = []
        # for drone in env.agents:
        #     save_pos.append(drone.pos.copy())

        for drone in env.agents:
            drone.load_best()
        for user in env.user_list:
            user.load_best()

        # def distance_3d(a, b, c): return np.sqrt(np.power(a, 2) + np.power(b, 2) + np.power(c, 2))
        #
        # for idx, drone in enumerate(env.agents):
        #     distance = distance_3d(drone.pos[0] - save_pos[idx][0],
        #                            drone.pos[1] - save_pos[idx][1],
        #                            drone.pos[2] - save_pos[idx][2])
        #     energy_iter_episode += distance / velocity * env.calc_power(velocity=velocity)
        #     power_iter_episode += env.calc_power(velocity=velocity)

        # Update metrics
        for user in env.user_list:
            user.connection = False
            user.index_dron = None
        for drone in env.agents:
            drone.users.clear()

        zero_actions = (np.ones(len(env.agents), dtype='int') * agents[0].actions.stop).tolist()
        reward, new_obs, done, _ = env.step(zero_actions)

        # Efficiency
        efficiency = np.asarray(efficiency)

        metric.update(len(env.user_list), env.calc_users_connected, env.agents, env.all_freq,
                      power_iter_episode, efficiency.mean(), time_tx_iter_episode / (iteration + 1),
                      energy_iter_episode)

        # Update observation spaces
        old_obs = new_obs.copy()
        if s_render:
            if episode % 10 == 0:
                env.render(filename=f'Episode_{episode + 1}.png')  # Render image environment

        if episode == n_episodes - 1:
            env.render(filename=f'Episode_{episode + 1}.png')  # Render image environment

        energy_iter_episode = 0
        power_iter_episode = 0
        time_tx_iter_episode = 0
        # env.move_user()  # User movement

    # All Episodes End
    best_pos_height = []
    for drone in env.agents:
        best_pos_height.append(drone.save_dict['save_position'][2])

    metric.extra_metric(f'{env.dir_sim}', env.agents, n_episodes)
    metric.save_height = best_pos_height.copy()
    metric.save_metric(run_i)
    show_iter(iter_x_episode, n_episodes, run_i)
    progress.show(time.time(), env.calc_users_connected, run_i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Name of the folder where the simulations will be saved.", default='Paper')
    parser.add_argument('-e', '--episodes', help="Number of the episodes.", type=int, default=10)
    parser.add_argument('-r', '--run', help="Number of the independent run.", type=int, default=1)
    parser.add_argument('-g', '--greedy', help="Use e-greedy or e-greedy with decay", type=float, default=0.5)
    parser.add_argument('-d', '--drone', help="Number of drones", type=int, default=10)
    parser.add_argument('-u', '--users', help="Number of users", type=int, default=200)
    parser.add_argument('-wu', '--weight_user', help='Weight for users', type=int, default=1)
    parser.add_argument('-wd', '--weight_drone', help='Weight for drones', type=int, default=1)
    parser.add_argument('-wc', '--weight_connection', help='Weight for connection', type=int, default=1)
    parser.add_argument('-f', '--frequency', help="List with operations frequencies", type=str, default="1e09")
    parser.add_argument('-t', '--thread', help='Number thread', type=int, default=1)
    parser.add_argument('-ls', '--length_step', help='Length step on coord Z', type=int, default=2)
    parser.add_argument('-s', '--show', help='Show render environment', type=int, default=0)
    parser.add_argument('-i', '--info', help="Name of an environment", default='cluster')
    args = parser.parse_args()

    weight_parser = {
        'Wu': args.weight_user,
        'Wd': args.weight_drone,
        'Wt': args.weight_connection
    }

    if args.greedy == 0:
        print(f'\nActive e-greedy decay')
    else:
        print(f'\nActive e-greedy {args.greedy}')

    np.seterr(divide='ignore', invalid='ignore')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    main_chapter = os.getcwd()
    try:
        os.chdir(args.name)
    except FileNotFoundError:
        os.mkdir(args.name)
        os.chdir(args.name)

    now_chapter = os.getcwd()
    copy(main_chapter + f'/mapa.pickle', now_chapter + f'/mapa.pickle')
    copy(main_chapter + f'/users_d_{args.info}.pickle', now_chapter + f'/users_d_{args.info}.pickle')

    Parallel(n_jobs=args.thread)(delayed(function_simulation)(i, args.episodes, args.greedy, args.drone, args.frequency,
                                                              args.users, weight_parser,
                                                              args.show, args.info, args.length_step)
                                 for i in range(args.run))

    fig_6(args.run)
    fig_11(args.run)
    fig_12(args.run)
    fig_status(args.run)
    fig_power(args.run)
    fig_energy(args.run)
    fig_efficiency(args.run)
    fig_time(args.run)
    fig_battery(args.run)
    fig_actions(args.run)
    fig_height(args.run)

    frames_path = 'Run_{i}/Episode_{j}.png'
    vid_name = 'Run_{i}/Run_{i}.mp4'

    if args.show:
        for i in range(args.run):
            with imageio.get_writer(vid_name.format(i=i), format='FFMPEG', mode='I', fps=1) as writer:
                writer.append_data(imageio.v2.imread(frames_path.format(i=i, j=0)))
                for j in range(args.episodes):
                    if j % 10 == 0 or j == args.episodes - 1:
                        writer.append_data(imageio.v2.imread(frames_path.format(i=i, j=j + 1)))

        for i in range(args.run):
            os.remove(frames_path.format(i=i, j=0))
            for j in range(args.episodes):
                if j % 10 == 0 or j == args.episodes - 1:
                    os.remove(frames_path.format(i=i, j=j + 1))

    lstFiles = []
    lstDir = os.walk(now_chapter)

    for root, dirs, files in lstDir:
        for file in files:
            (filename, extension) = os.path.splitext(file)
            if extension == ".npz":
                lstFiles.append(filename + extension)

    for file in lstFiles:
        os.remove(file)