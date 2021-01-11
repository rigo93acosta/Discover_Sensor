import numpy as np
import matplotlib.pyplot as plt


class Metric:

    def __init__(self, ind_run, ind_episodes, actions):

        self.total_run = ind_run
        self.episodes = ind_episodes
        self.save_reward = []
        self.save_backhaul = []
        self.save_backhaul_1 = []
        self.save_status = []
        self.save_power = []
        self.save_energy = []
        self.save_efficiency = []
        self.save_time_tx = []
        self.save_battery = []
        self.actions_name = actions
        self.save_actions = []

    @staticmethod
    def calc_drone_ran(drones):
        """
        Calculate % occupation user RAN

        Args:
            drones:

        Returns:
            object:
        """
        total_active = 0
        actual = 0
        for drone in drones:
            if drone.status_tx:
                actual += drone.get_len_users
                total_active += 1

        total_ran = total_active * 50
        return actual * 100 / total_ran

    @staticmethod
    def calc_backhaul(drones, flag_backhaul):
        """
        Calculate average backhaul

        Args:
            drones:
            flag_backhaul:

        Returns:
            float:
        """
        total_backhaul = []
        for drone in drones:
            if drone.status_tx:
                total_backhaul.append(drone.actual_capacity)
        if flag_backhaul:   # Backhaul per drone
            return float(np.mean(total_backhaul) / 1e06)
        else:   # Backhaul global
            return float(np.sum(total_backhaul) / 1e06)

    @staticmethod
    def _calc_status(drones):

        val_temp = 0
        for dron in drones:
            if dron.status_tx:
                val_temp += 1

        return val_temp

    def update(self, reward_max, reward, drones, frequencies,
               power, efficiency, time_tx, energy):
        """
        Update metrics simulation
        Args:
            time_tx: Transmission Time
            reward_max: Max Reward
            reward: Reward of the best position
            drones: List drones
            frequencies: List frequencies in use
            power: Power total
            efficiency: Efficiency
            energy: Consumption Energy
        """
        self.save_reward.append((reward_max - reward) * 100 / reward_max)
        self.save_backhaul.append(self.calc_backhaul(drones, True) / len(frequencies))
        self.save_backhaul_1.append(self.calc_backhaul(drones, False) / len(frequencies))
        self.save_status.append(self._calc_status(drones))
        self.save_power.append(power)
        self.save_energy.append(energy)
        self.save_efficiency.append(efficiency)
        self.save_time_tx.append(time_tx)
        self.save_battery.append(np.asarray([(drone.battery*100)/drone.max_battery for drone in drones]).mean())

    def save_metric(self, run_i=0):
        """
        Save metrics simulation for independent run

        Args:
        run_i: Run in action
        """
        np.savez(f'Run_{run_i}', data=self.save_reward)
        np.savez(f'Run_backhaul_drone{run_i}', data=self.save_backhaul)
        np.savez(f'Run_backhaul_global{run_i}', data=self.save_backhaul_1)
        np.savez(f'Run_status_{run_i}', data=self.save_status)
        np.savez(f'Run_power_{run_i}', data=self.save_power)
        np.savez(f'Run_energy_{run_i}', data=self.save_energy)
        np.savez(f'Run_efficiency_{run_i}', data=self.save_efficiency)
        np.savez(f'Run_time_{run_i}', data=self.save_time_tx)
        np.savez(f'Run_battery_{run_i}', data=self.save_battery)
        np.savez(f'Run_actions_{run_i}', data=self.save_actions)

    def extra_metric(self, chapter, drones, n_episodes):

        temp = []
        for visual_index, drone in enumerate(drones):
            counts, bins = np.histogram(drone.shift, bins=len(self.actions_name))
            if counts[6] != 0:
                counts[6] -= n_episodes
            temp.append(counts)

        temp = np.ceil(np.mean(np.array(temp), axis=0))
        self.save_actions = temp.tolist()
        _, ax = plt.subplots()
        ax.bar(self.actions_name, temp)
        ax.set_xlabel(f'Actions')
        ax.set_ylabel(f'Repeat')
        ax.set_title(f'Average actions')
        filename = chapter + '/' + f'Ave_action'
        plt.savefig(f'{filename}.png', dpi=200)
        plt.close()
