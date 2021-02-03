from enum import IntEnum
import numpy as np
import gym
from collections import namedtuple
import random


class Drone:
    """
    Class Drone
    """

    class DefaultActions(IntEnum):
        up = 0  # Drone move up
        down = 1  # Drone move down
        right = 2  # Drone move right
        left = 3  # Drone move left
        forw = 4  # Drone move forward
        back = 5  # Drone move backward
        stop = 6  # Drone not move
        # on = 7  # Drone on
        # off = 8  # Drone off

    def __init__(self, frequency, step):
        self.pos = []
        self.name = 'Drone'
        self.capacity = 37.5e6 * (1 - 0.1 * np.random.rand())
        self.actual_capacity = 0
        self.max_capacity = 25
        self.actions = self.DefaultActions
        self.users = []
        self.shift = []
        self.altitude = []
        self.step_amplitude_z = 2
        self.flag_step = step
        self.battery = 360e03
        self.max_battery = 360e03  # 360 KJ Battery Energy
        self.distance = namedtuple('Distance', ['horizontal', 'vertical'])

        self.status_tx = True  # Transmission is enable(True) or disable(False)
        self.freq_tx = random.choice(frequency)
        self.all_freq = frequency  # Frequency transmission available

        self.save_dict = {'save_users': [], 'save_freq': 0,
                          'save_position': [], 'save_status': True}

        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.observation_space = gym.spaces.Dict(
            {'position': gym.spaces.MultiDiscrete([self.space_map[0].shape[0],
                                                   self.space_map[1].shape[0],
                                                   self.space_map[2].shape[0]]),
             'frequencies': gym.spaces.Discrete(len(self.all_freq)),
             'tx_status': gym.spaces.Discrete(2)
             }
        )

        self.q_table = np.zeros((self.observation_space['position'].nvec[0],
                                 self.observation_space['position'].nvec[1],
                                 self.observation_space['position'].nvec[2],
                                 self.observation_space['tx_status'].n,
                                 self.observation_space['frequencies'].n,
                                 self.action_space.n))

    def __repr__(self):
        return f'{self.name}(Position({self.pos[0]}, {self.pos[1]}, {self.pos[2]}), {len(self.users)} Users Connected' \
               f') and F_tx:{self.freq_tx} '

    @property
    def space_map(self):
        """
        All positions drone on map

        Returns:
            List values
        """
        all_pos_x = np.arange(0, 201, 20)
        all_pos_y = np.arange(0, 201, 20)
        if self.flag_step == 1:
            all_pos_z = np.arange(10, 101, 10)
        elif self.flag_step == 2:
            all_pos_z = np.arange(20, 101, 20)

        return [all_pos_x, all_pos_y, all_pos_z]

    @property
    def position(self):
        return self.pos

    @position.setter
    def position(self, pos_list):
        self.pos = pos_list

    @property
    def get_len_users(self):
        return len(self.users)

    def choice_action(self, obs_state, epsilon):
        """
        Drone select action
        Args:
            obs_state: drone actual state
            epsilon: epsilon-greedy

        Returns:
        Select action
        """
        if np.random.uniform(0, 1) < epsilon:  # Random
            a_selected = np.random.randint(self.action_space.n)
        else:  # Knowledge
            val = self.q_table[obs_state[0], obs_state[1], obs_state[2], obs_state[3], obs_state[4]]
            a_selected = np.random.choice([action_ for action_, value_ in enumerate(val) if value_ == np.max(val)])

        action_correct = self.validate_action(a_selected, obs_state)

        # if not self.status_tx:
        #     if action_correct != self.actions.stop and action_correct != self.actions.on:
        #         action_correct = 6
        #     if a_selected != self.actions.stop and a_selected != self.actions.on:
        #         a_selected = 6

        return action_correct, a_selected

    def validate_action(self, action, now_state):  # Only if drone are active
        """
        Validation action drone
        Args:
            action: action selected
            now_state: actual state drone

        Returns:
        Correct action
        """
        action_back = action
        max_space_z = self.observation_space['position'].nvec[2] - 1
        max_space_y = self.observation_space['position'].nvec[1] - 1
        max_space_x = self.observation_space['position'].nvec[0] - 1

        if action == self.actions.up:
            if now_state[2] == max_space_z:
                action_back = 6

        elif action == self.actions.down:
            if self.step_amplitude_z == 1:
                if now_state[2] == 0:
                    action_back = 6
            elif self.step_amplitude_z == 2:
                if now_state[2] == 1:
                    action_back = 6

        elif action == self.actions.right:
            if now_state[0] == max_space_x:
                action_back = 6

        elif action == self.actions.left:
            if now_state[0] == 0:
                action_back = 6

        elif action == self.actions.forw:
            if now_state[1] == max_space_y:
                action_back = 6

        elif action == self.actions.back:
            if now_state[1] == 0:
                action_back = 6

        elif action == self.actions.stop:
            action_back = 6

        # elif action == self.actions.off:    # Turn off
        #     action_back = action
        #
        # elif action == self.actions.on:  # Turn on
        #     action_back = action

        # elif action == self.actions.change:  # Changed frequency
        #     action_back = action

        return action_back

    def learn(self, old_state, new_state, values):
        """
        Drone learn interaction with environment
        Args:
            old_state: Previous state drone
            new_state: Actual state drone
            values: [0]-Learning Rate [1]-Discount factor [2]-Reward scenario [3]-Old action
        """
        max_future = np.max(self.q_table[new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]])
        actual_value_state = self.q_table[old_state[0], old_state[1], old_state[2],
                                          old_state[3], old_state[4], values[3]]
        new_q = (1 - values[0]) * actual_value_state + values[0] * (values[2] + values[1] * max_future)
        self.q_table[old_state[0], old_state[1], old_state[2], old_state[3], old_state[4], values[3]] = new_q

    def action_step(self, value):
        """
        Step action drone
        Args:
            value: action choice for algorithm
        """
        if value == self.actions.stop:
            self.pos[0] += 0
            self.pos[1] += 0
            self.distance.horizontal = 0
            self.distance.vertical = 0

        elif value == self.actions.up:
            self.pos[2] += 10 * self.step_amplitude_z
            self.distance.horizontal = 0
            self.distance.vertical = 10 * self.step_amplitude_z

        elif value == self.actions.down:
            self.pos[2] -= 10 * self.step_amplitude_z
            self.distance.horizontal = 0
            self.distance.vertical = -10 * self.step_amplitude_z

        elif value == self.actions.right:
            self.pos[0] += 20
            self.distance.horizontal = 20
            self.distance.vertical = 0

        elif value == self.actions.left:
            self.pos[0] -= 20
            self.distance.horizontal = 20
            self.distance.vertical = 0

        elif value == self.actions.forw:
            self.pos[1] += 20
            self.distance.horizontal = 20
            self.distance.vertical = 0

        elif value == self.actions.back:
            self.pos[1] -= 20
            self.distance.horizontal = 20
            self.distance.vertical = 0

        # elif value == self.actions.on:
        #     self.status_tx = True
        #     self.distance.horizontal = 0
        #     self.distance.vertical = 0
        #
        # elif value == self.actions.off:
        #     self.status_tx = False
        #     self.distance.horizontal = 0
        #     self.distance.vertical = 0

        # elif value == self.actions.change:
        #     index = self.all_freq.index(self.freq_tx)
        #     index += 1
        #     index %= len(self.all_freq)
        #     self.freq_tx = self.all_freq[index]
        #     self.distance.horizontal = 0
        #     self.distance.vertical = 0
        self.altitude.append(self.pos[2])
        self.shift.append(value)

    def save_best(self):
        """
        Save best scenario
        """
        # self.save_dict['save_users'].clear()
        self.save_dict['save_users'] = self.users.copy()
        # self.save_dict['save_position'].clear()
        self.save_dict['save_position'] = self.pos.copy()
        self.save_dict['save_status'] = self.status_tx
        self.save_dict['save_freq'] = self.freq_tx

    def load_best(self):
        """
        Load best scenario
        """
        # self.users.clear()
        self.users = self.save_dict['save_users'].copy()
        # self.pos.clear()
        self.pos = self.save_dict['save_position'].copy()
        self.status_tx = self.save_dict['save_status']
        self.freq_tx = self.save_dict['save_freq']
