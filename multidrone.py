import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import gym
from collections import namedtuple
from e_model import EnergyModel

matplotlib.use("Agg")
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'
# plt.rcParams['ztick.labelsize'] = 18

with open('mapa.pickle', 'rb') as f:
    info = pickle.load(f)


class MultiDroneEnv(gym.Env):

    def __init__(self, agents, n_users=300, frequency=None, weight=1, n_run=0):

        if frequency is None:
            frequency = [1e09]

        self.terreno = info['terreno']  # Need to allocate users in the map
        self.agents = agents
        self.total_user = n_users
        self.output_chapter = f'run'
        self.user_list = []
        self.all_freq = frequency
        self.reward_range = (0, self.get_max_reward)
        self.val_a, self.val_b = self.values_a_b
        self.epsilon = None
        self.weight = weight
        self.val_velocity = 0
        self._threshold = -3
        self.model_dict = namedtuple('Model', 'energy power time_tx efficiency')
        self.simulation_run = n_run
        self.info = 'cluster'

    def reset(self):
        """
        Reset or Init Environment

        Returns:
            States drones
        """
        # temp_position_user = self._create_users
        with open(f'users_d_{self.info}.pickle', 'rb') as file:
            user_distribution = pickle.load(file)

        temp_position_user = user_distribution[self.simulation_run]
        for index_user in range(self.total_user):
            temp_user = User(name='User_' + str(index_user + 1), init_req_th=0)
            temp_user.position = [temp_position_user[index_user][0], temp_position_user[index_user][1], 0]
            self.user_list.append(temp_user)

        initial_altitude = [10, 50, 100]
        for id_drone, drone in enumerate(self.agents):
            # temp_state = drone.observation_space.sample()
            drone.position = [10,
                              10,
                              np.random.choice(initial_altitude, 1)[0]]

        return self._get_obs

    def step(self, array_actions):
        """
        Step Environment

        Args:
            array_actions: Actions for drones

        Returns:
            List: [Reward scenario, States drones, Done task, Info]
        """
        frequency_drone = []
        status_drone = []
        for id_d, dron in enumerate(self.agents):
            dron.action_step(array_actions[id_d])  # Move Users
            frequency_drone.append(dron.freq_tx)
            status_drone.append(dron.status_tx)

        user_array = np.asarray([user.position for user in self.user_list])
        dron_array = np.asarray([dron.position for dron in self.agents])
        dist_dron_r = np.zeros((user_array.shape[0], dron_array.shape[0]))

        for idx_dron in range(len(self.agents)):
            dist_dron_r[:, idx_dron] = np.sqrt(np.sum(np.square(user_array[:, :2] - dron_array[idx_dron, :2]), axis=1))

        sinr_db = self._interplay(dron_array, dist_dron_r, frequency_drone, status_drone)  # Calculate SINR
        self._user_to_drone(sinr_db)  # Connected Users to Drone and Users Throughput
        self.calc_backhaul()  # Calculated Backhaul
        info_env = None
        var_1, var_2, var_3 = self.model_energy(self.val_velocity)
        self.model_dict.efficiency = self._calc_efficiency(var_1)
        self.model_dict.energy, self.model_dict.power, self.model_dict.time_tx = var_1.sum(), var_2, var_3
        reward = self.calc_reward(self._search_worse_user())  # Rewards
        done = False
        if self.calc_users_connected == self.get_max_reward:
            done = True

        return reward, self._get_obs, done, info_env

    def render(self, filename=f'Env.png'):
        """
        Render image environment
        Args:
            filename: Render filename
        """
        from PIL import Image

        fig = plt.figure(figsize=(15.04, 15.04), dpi=100)
        canvas = FigureCanvasAgg(fig)

        dron_array = np.asarray([dron.position for dron in self.agents])
        user_array = np.asarray([user.position for user in self.user_list])
        split_array = np.hsplit(dron_array, 3)
        dron_x, dron_y, dron_z = split_array[0], split_array[1], split_array[2]
        split_array = np.hsplit(user_array, 3)
        user_x, user_y, user_z = split_array[0], split_array[1], split_array[2]

        ax = fig.gca(projection='3d')

        limits = np.array([[0, 200],
                           [0, 200],
                           [10, 101]])

        frequency_index = []
        for drone in self.agents:
            frequency_index.append(self.all_freq.index(drone.freq_tx))

        for id_d, dron in enumerate(self.agents):
            if dron.status_tx:
                c1 = self.get_circle(x=[dron.pos[0], dron.pos[1], 0], r=dron.pos[2] * np.tan(np.deg2rad(60 / 2)))
                _, c3 = self.cut_circle(c1, limits=limits)
                if len(c3) != 0:
                    ax.plot(*c3.T, c=self._set_colors(id_d), lw=3, ls='-')

        lstconected = []
        users_connect = 0
        labels_drone = []
        for id_d, dron in enumerate(self.agents):
            if len(dron.users) != 0 and dron.status_tx:
                lstconected.extend(dron.users)
                ax.scatter(user_x[dron.users], user_y[dron.users],
                           s=40, marker='o', color=self._set_colors(id_d), depthshade=False)

            values_text = self._freq_string(dron.freq_tx)
            marker_tips = ['^', 'X', 'D', 'o', 's', '*']
            if dron.status_tx:
                legend_text = 'A: {}m\nF: {} U: {}'.format(dron.position[2], values_text, dron.get_len_users)
            else:
                legend_text = 'A: {}m\nOFF'.format(dron.position[2])

            if dron.status_tx:
                label_text = ax.scatter(dron_x[id_d], dron_y[id_d], dron_z[id_d], s=150,
                                        marker=marker_tips[frequency_index[id_d]], color=self._set_colors(id_d),
                                        label=legend_text, depthshade=False)
            else:
                label_text = ax.scatter(dron_x[id_d], dron_y[id_d], dron_z[id_d], s=150,
                                        marker=marker_tips[frequency_index[id_d]], color='k',
                                        label=legend_text, depthshade=False)

            labels_drone.append(label_text)
            users_connect += dron.get_len_users

        desconectados = set(range(self.total_user))
        lstconected = set(lstconected)
        desconectados = list(desconectados.difference(lstconected))
        ax.scatter(user_x[desconectados], user_y[desconectados], s=40, marker='x', color='k', depthshade=False)

        try:
            os.mkdir(self.output_chapter)
        except:
            pass

        enable_drones = 0
        for dron in self.agents:
            if dron.status_tx:
                enable_drones += 1

        dir_file = self.output_chapter + '/' + filename
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_xlim3d(0, info['L'])
        ax.set_ylim3d(0, info['L'])
        ax.set_zlim3d(0, 101)
        ax.view_init(elev=25, azim=-45)
        ax.set_title(f'User connect {users_connect} - Active drones {enable_drones}')
        fig.legend(loc=7, prop={'weight': 'bold', 'size': 14})
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        canvas.draw()

        # array_fig = np.array(fig.canvas.renderer.buffer_rgba())

        s, (width, height) = canvas.print_to_buffer()
        array_fig = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        im = Image.fromarray(array_fig)
        # im = Image.frombytes("RGBA", (width, height), s)
        im.save(f'{dir_file}')
        plt.close()

    @property
    def _set_colors(self):
        return plt.cm.get_cmap('jet', len(self.agents))

    @staticmethod
    def get_circle(x, r, n=360):

        q = np.linspace(start=-np.pi, stop=np.pi, num=n)

        c_data = np.empty((n, len(x)))
        c_data[:, 0] = np.cos(q) * r + x[0]
        c_data[:, 1] = np.sin(q) * r + x[1]
        if len(x) == 3:
            c_data[:, 2] = x[2]

        return c_data

    @staticmethod
    def cut_circle(c_data, limits):
        idx_keep_x = np.logical_and(limits[0, 0] < c_data[:, 0], c_data[:, 0] < limits[0, 1])
        idx_keep_y = np.logical_and(limits[1, 0] < c_data[:, 1], c_data[:, 1] < limits[1, 1])
        idx_keep = np.logical_and(idx_keep_x, idx_keep_y)

        c_2 = c_data[idx_keep, :]

        c_2_help = np.concatenate([c_2, c_2[:1, :]])
        if len(c_2_help) != 0:
            i_jump = np.argmax(np.linalg.norm(np.diff(c_2_help[:, :2], axis=0), axis=-1))
            c_3 = np.roll(c_2, -(i_jump + 1), axis=0)
        else:
            c_2, c_3 = [], []

        return c_2, c_3

    @property
    def dir_sim(self):
        return self.output_chapter

    @dir_sim.setter
    def dir_sim(self, value):
        self.output_chapter = value

    @property
    def get_max_reward(self):
        return self.total_user

    @property
    def get_epsilon(self):
        return self.epsilon

    @get_epsilon.setter
    def get_epsilon(self, value):
        self.epsilon = value

    @property
    def _get_obs(self):
        """
        Search the state for drones

        Returns:
            List states drones
        """

        def search_state(x, list_pos):
            return list_pos.index(x)

        temp_obs = []

        for dron in self.agents:
            temp_position = dron.position
            val1 = search_state(temp_position[0], list(dron.space_map[0]))
            val2 = search_state(temp_position[1], list(dron.space_map[1]))
            val3 = search_state(temp_position[2], list(dron.space_map[2]))
            if dron.status_tx:
                val4 = 1  # Enable Tx
            else:
                val4 = 0  # Disable Tx
            val5 = self.all_freq.index(dron.freq_tx)
            temp_obs.append([val1, val2, val3, val4, val5])

        return temp_obs

    @property
    def values_a_b(self):
        """
        Calculate values for simulation relationship with LOS o not LOS

        Returns:
            Constants value A and B
        """
        c_a = [9.34e-01, 2.30e-01, -2.25e-03, 1.86e-05, 1.97e-02, 2.44e-03, 6.58e-06,
               0, -1.24e-04, -3.34e-06, 0, 0, 2.73e-07, 0, 0, 0]
        c_b = [1.17e-00, -7.56e-02, 1.98e-03, -1.78e-05, -5.79e-03, 1.81e-04, -1.65e-06,
               0, 1.73e-05, -2.02e-07, 0, 0, -2.00e-08, 0, 0, 0]

        c_a = np.asarray(c_a).reshape(4, 4)
        c_b = np.asarray(c_b).reshape(4, 4)

        z_a = []
        z_b = []

        for j in range(4):
            for i in range(4 - j):
                z_a.append(c_a[i, j] * np.power(info['alpha'] * info['beta'], i) * np.power(info['gamma'], j))

        for j in range(4):
            for i in range(4 - j):
                z_b.append(c_b[i, j] * np.power(info['alpha'] * info['beta'], i) * np.power(info['gamma'], j))

        return sum(z_a), sum(z_b)

    @staticmethod
    def _loss_path_average(dr, dh, data_a, data_b, f_tx):
        """
        Calculate loss free space + LOS
        Args:
            dr: Distance users-drones in 2D
            dh: Heights drones
            data_a: Values for LOS
            data_b: Values for LOS

        Returns:
            Loss Path between users-drones
        """

        # Calculate for a frequency 1GHz
        np.seterr(divide='ignore', invalid='ignore')
        val_division = np.true_divide(dh, dr)
        term_a = 1 + data_a * np.exp(-data_b * (np.rad2deg(np.arctan(val_division)) - data_a))
        term_c = 1
        term_d = 20
        term_b = 10 * np.log10(np.square(dh) + np.square(dr)) + 20 * np.log10(f_tx) + 20 * np.log10(4 * np.pi / 3e08)

        return term_b + term_c

    def _calc_rsrp(self, drones, distance_2d, f_tx, status_tx):
        """
        Args:
            drones: Position all drones in 3D
            distance_2d: Distance users-drones in 2D

        Returns:
            Loss without noise
        """

        # Equation 7 in the paper.
        eirp = -3  # -3 dBW
        dron_rc = drones[:, 2] * np.tan(np.deg2rad(60 / 2))  # Radio coverage all drones
        dist_drones = distance_2d[:, :len(self.agents)]
        drones_altura = drones[:len(self.agents), 2]
        result_loss = eirp - self._loss_path_average(dist_drones, drones_altura, self.val_a, self.val_b, f_tx)
        table_ok_cob = distance_2d[:, :len(self.agents)] <= dron_rc[:len(self.agents)]
        user_false, dron_false = np.where(table_ok_cob == False)
        result_loss[user_false, dron_false] = -530  # Equal to -500dBm

        for index, status in enumerate(status_tx):
            if not status:
                result_loss[:, index] = -2000

        return result_loss

    def _interplay(self, drones, dr, f_tx, status_tx):
        """
        Result agents-environment interplay
        Args:
            drones: Position drone array
            dr: Distance users-drones in 2D

        Returns:
            SINR array
        """

        table_rsrp_db = self._calc_rsrp(drones, dr, f_tx, status_tx)
        table_rsrp_linear = np.power(10, table_rsrp_db / 10)

        # Equal to sum at denominator for equation 8
        table_rsrp_inv_linear = np.zeros((self.total_user, len(self.agents)))
        for j in range(table_rsrp_inv_linear.shape[1]):
            f_tx_e = f_tx[j]
            index = np.where(np.asarray(f_tx) == f_tx_e)
            index = index[0]
            for k in index:
                if k != j:
                    table_rsrp_inv_linear[:, j] += table_rsrp_linear[:, k]

        # Additive White Gaussian Noise
        awgn_linear = np.power(10, -120 / 10) / 1000

        return 10 * np.log10(table_rsrp_linear) - 10 * np.log10(awgn_linear + table_rsrp_inv_linear)

    def _user_to_drone(self, table_sinr):
        """
        Assign users to drone
        Args:
            table_sinr: SINR array
        """

        # Function lambda to calculate throughput
        def calc_thr(x):
            return 180e03 * np.log2(1 + np.power(10, x / 10))

        for idx_user, user in enumerate(self.user_list):
            if user.connection:
                idx_drone = user.index_dron
                if table_sinr[idx_user, idx_drone] >= self._threshold:
                    user.throughput = 1.3 * calc_thr(table_sinr[idx_user, idx_drone])
                else:
                    user.throughput = 0
                    user.connection = False
                    self.agents[idx_drone].users.remove(idx_user)
                    user.index_dron = None

        number_user = np.arange(0, self.total_user)
        for idx_user in number_user:
            if not self.user_list[idx_user].connection:
                order_drone = np.flip(np.argsort(table_sinr[idx_user]))
                for idx_drone in order_drone:
                    if table_sinr[idx_user, idx_drone] >= self._threshold:
                        if idx_user not in self.agents[idx_drone].users:
                            if len(self.agents[idx_drone].users) < self.agents[idx_drone].max_capacity:
                                self.agents[idx_drone].users.append(idx_user)
                                self.user_list[idx_user].connection = True
                                self.user_list[idx_user].throughput = 1.3 * calc_thr(table_sinr[idx_user, idx_drone])
                                self.user_list[idx_user].index_dron = idx_drone
                                break

    def calc_reward(self, worse_th):
        """
        Calculate Reward

        Returns:
            Reward scenario
        """
        r_min = 180e03 * np.log2(1 + np.power(10, self._threshold / 10))  # Minimum throughput threshold
        return (self.weight['Wu'] * self.calc_users_connected) + (self.weight['Wt'] * ((worse_th - r_min) / r_min))

    @property
    def calc_users_connected(self):
        """
        Calculate Reward

        Returns:
            Reward scenario
        """
        iteration_reward = 0
        for dron in self.agents:
            val = np.asarray(dron.users)
            if len(val) == 0:
                iteration_reward += 0
            else:
                iteration_reward += val.shape[0]

        return iteration_reward

    def calc_backhaul(self):
        """
        Calculate average backhaul
        """
        for select_dron in self.agents:
            val_back = 0
            for user_in_uav in select_dron.users:
                val_back += self.user_list[user_in_uav].throughput

            # if val_back > self.dron_list[select_dron].capacity:
            #     new_capacity = 0.9 * self.dron_list[select_dron].capacity
            #     dif = new_capacity - val_back
            #     assign = dif / len(self.dron_list[select_dron].users)
            #     for user_in_uav in self.dron_list[select_dron].users:
            #         now_assign = self.user_list[user_in_uav].throughput
            #         self.user_list[user_in_uav].set_th(now_assign + assign)
            #
            # val_back = 0
            # for user_in_uav in self.dron_list[select_dron].users:
            #     val_back += self.user_list[user_in_uav].throughput
            select_dron.actual_capacity = val_back

    def move_user(self):
        """
        Function to move users
        """

        # Ref -> Efficient 3D Aerial Base Station Placement Considering Users Mobility by RL
        mov_angle = {'min': 0, 'max': 2 * np.pi}

        # mov_speed = {'min': 0, 'max': 1.3}  # Value is 1.3 m/s
        # mov_time = 2  # Values is a seconds

        def delta_y(angle, distance):
            return int(np.round(distance * np.sin(angle)))

        def delta_x(angle, distance):
            return int(np.round(distance * np.cos(angle)))

        def value_random(v_min, v_max):
            return np.random.uniform(v_min, v_max)

        for user in self.user_list:
            name = user.name
            name = name.split('_')
            name = name[0]

            if name == 'User':
                change_pos = False
                while not change_pos:
                    sum_y = delta_y(value_random(mov_angle['min'], mov_angle['max']), 1)
                    sum_x = delta_x(value_random(mov_angle['min'], mov_angle['max']), 1)

                    past_x, past_y = user.position[0], user.position[1]
                    user.action_step([sum_x, sum_y])
                    now_x, now_y = user.position[0], user.position[1]
                    if self.terreno[now_x, now_y] == 0:
                        self.terreno[past_x, past_y] = 0
                        self.terreno[now_x, now_y] = 200
                        change_pos = True

    @staticmethod
    def _freq_string(frequency):

        def func_text(x): return "{:2.1E}".format(x)

        values_dict = {'03': 'KHz', '06': 'MHz', '09': 'GHz'}
        out_values = func_text(frequency).split('E+')
        out_values[1] = values_dict[out_values[1]]
        return out_values[0]  # + out_values[1]

    # Users deployment
    @property
    def _create_users(self):
        """
        Allocate user in environment
        Returns:
            Users position in map
        """
        temp_user = []
        # Random users position
        for _ in range(self.total_user):  # Number of the users
            position_user_error = False
            while not position_user_error:
                user_x = int(np.random.randint(0, info['L'], 1))
                user_y = int(np.random.randint(0, info['L'], 1))
                if self.terreno[user_x, user_y] > 0:
                    position_user_error = False
                else:
                    self.terreno[user_x, user_y] = 200
                    temp_user.append((user_x, user_y))
                    position_user_error = True

        return temp_user

    def model_energy(self, velocity):

        # Energy Model Iteration
        power = np.zeros(shape=len(self.agents))
        energy = np.zeros(shape=len(self.agents))

        distance = np.zeros(shape=len(self.agents), dtype=int)
        drone_vel = np.zeros(shape=len(self.agents))

        math_model_energy = EnergyModel(velocity=velocity, rotors=4)

        def val_angle(val_h, val_v):
            return np.arctan(np.true_divide(val_h, val_v))

        power_hovering = math_model_energy.power_hover
        flag_ind_horizontal = True
        flag_ind_height = 0

        for index, drone in enumerate(self.agents):

            if drone.distance.horizontal != 0:
                distance[index] = np.abs(drone.distance.horizontal)
                drone_vel[index] = velocity * np.sin(val_angle(drone.distance.horizontal, drone.distance.vertical))
                flag_ind_horizontal = True
            elif drone.distance.vertical != 0:
                distance[index] = np.abs(drone.distance.vertical)
                drone_vel[index] = velocity * np.cos(val_angle(drone.distance.horizontal, drone.distance.vertical))
                flag_ind_height = 1 if drone.distance.vertical >= 0 else 0
                flag_ind_horizontal = False
            else:
                distance[index] = 0

            if distance[index] == 0:
                power[index] = power_hovering
            else:
                if flag_ind_horizontal:
                    power[index] = math_model_energy.power_horizontal(math_model_energy.power_p(drone_vel[index]),
                                                                      math_model_energy.power_i(drone_vel[index]))
                elif flag_ind_height:
                    power[index] = math_model_energy.power_a_d(drone_vel[index], True)
                else:
                    power[index] = math_model_energy.power_a_d(drone_vel[index], False)

        # velocity_hover = math_model_energy.velocity_hover
        time_shift = np.true_divide(distance, velocity)

        t_mov = 10  # Movement Time
        t_r = 1  # Handshake Time
        t_c = 0  # Exchange Time
        time_iteration = t_mov + t_c + t_r

        for index, time in enumerate(time_shift):
            if time == 0:
                energy[index] = time_iteration * power_hovering
            elif time == 5:
                energy[index] = time * (power_hovering + power[index]) + t_r * power_hovering
                power[index] += power_hovering
            elif time == 10:
                energy[index] = time * power[index] + t_r * power_hovering

            if self.agents[index].battery >= .1 * self.agents[index].max_battery:
                self.agents[index].battery -= energy[index]
            # else:
            #     self.agents[index].status_tx = False

        time_tx = time_iteration - time_shift - t_r
        return energy, power.sum(), np.asarray(time_tx).mean()

    def _calc_efficiency(self, list_energy):

        efficiency = np.zeros(shape=len(self.agents))
        r_min = 180e03 * np.log2(1 + np.power(10, self._threshold / 10))  # Minimum throughput threshold
        for idx_agent, agent in enumerate(self.agents):
            t_g = 0
            u_t = agent.get_len_users
            for idx_user in agent.users:
                t_g += self.user_list[idx_user].throughput

            efficiency[idx_agent] = ((self.weight['Wu'] * u_t * r_min) +
                                     (self.weight['Wt'] * (t_g - u_t * r_min))) / list_energy[idx_agent]

        return efficiency.mean()

    def _search_worse_user(self):

        dict_user = {}
        for idx, user in enumerate(self.user_list):
            if user.throughput != 0:
                dict_user[idx] = user.throughput
        sorted_dict = {k: v for k, v in sorted(dict_user.items(), key=lambda item: item[1])}  # Order dict
        idx = list(sorted_dict.values())
        if len(idx) == 0:
            return 0
        else:
            return idx[0]


class User:
    """
    Class User
    """

    def __init__(self, name='User', init_req_th=0):
        self.pos = []
        self.name = name
        self.req_th = np.power(10, init_req_th / 10)
        self.th_allocate = 0
        self.connection = False
        self.index_dron = None
        self.save_status = False  # Save status connection
        self.save_index = None  # Save id dron TRUE connection

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), Connection:{self.connection})'

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, list_pos):
        self.pos = list_pos

    def action_step(self, mov):
        """
        User action step
        Args:
            mov: array with action move user
        """
        if (self.pos[0] + mov[0] >= 0) and (self.pos[0] + mov[0]) < 500:
            self.pos[0] += mov[0]
        else:
            self.pos[0] += 0

        if (self.pos[1] + mov[1] >= 0) and (self.pos[1] + mov[1]) < 500:
            self.pos[1] += mov[1]
        else:
            self.pos[1] += 0

    @property
    def throughput(self):
        return self.th_allocate

    @throughput.setter
    def throughput(self, valor):
        self.th_allocate = valor

    def save_best(self):
        """
        Save best scenario
        """
        self.save_index = self.index_dron
        self.save_status = self.connection

    def load_best(self):
        """
        Load best scenario
        """
        self.index_dron = self.save_index
        self.connection = self.save_status
