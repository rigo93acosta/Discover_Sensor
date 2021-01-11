import numpy as np


class EnergyModel:

    def __init__(self, velocity, rotors=4):
        self.rotors = rotors
        self.velocity = velocity

        self.thrust = 20  # G in article
        self.rho = 1.2  # A fluid density of the air kg/m^3
        self.beta = 0.25  # A Rotor radius (m)
        self.cd0 = 0.025  # drag coefficient
        self.s_capital = 0.192  # Reference area (m^2)
        self.cb = 0.022  # Rotor chord (m)
        self.vel_ang = 20  # rad/s
        self.val_a = np.pi * np.power(self.beta, 2)  # Rotor disk area m^2
        self.n_blades = 2

    @property
    def power_hover(self):
        num = np.power(self.thrust, 3 / 2)
        den = np.sqrt(2 * self.rho * self.val_a * np.power(self.rotors, 3))
        return num / den

    @property
    def velocity_hover(self):
        return np.sqrt(self.thrust / (2 * self.rho * self.val_a * self.rotors))

    def power_p(self, vel_h):
        """
        Parasitic Power
        """
        # Part 1
        part_1 = 0.5 * self.rho * self.cd0 * self.s_capital * np.power(vel_h, 3)
        # Part 2
        part_2_a = (np.pi / 4) * self.rotors * self.n_blades * self.rho * self.cb * self.cd0 * self.beta
        part_2_b = np.power((self.vel_ang * self.beta), 3)
        # Part 3
        part_3 = 1 + 3 * np.power((vel_h / (self.vel_ang * self.beta)), 2)

        return part_1 + (part_2_a * part_2_b) * part_3

    def power_i(self, vel_h):
        """
        Induced power
        """
        val_lambda_part_2 = self.thrust / (self.rho * self.val_a * self.rotors)
        val_lambda = np.sqrt(np.power(vel_h, 4) + np.power(val_lambda_part_2, 2))
        num = val_lambda - np.power(vel_h, 2)
        return (self.thrust / self.rotors) * np.sqrt(num / 2)

    @staticmethod
    def power_horizontal(power_p, power_i):
        return power_p + power_i

    def power_a_d(self, vel, flag_a=True):
        """
        Ascending power and descending power

        vel: ascending velocity or descending velocity
        flag_a: True --> ascending
                False --> descending
        """
        part_1 = (self.thrust / (2 * self.rotors)) * vel

        part_2_a = self.thrust / (2 * self.rotors)

        part_2_b_a = np.power(vel, 2)

        part_2_b_num = 2 * self.thrust
        part_2_b_den = self.rho * self.val_a * self.rotors

        if flag_a:
            part_2_b = np.sqrt(part_2_b_a + part_2_b_num / part_2_b_den)
        else:
            if (part_2_b_a - part_2_b_num / part_2_b_den) < 0:
                part_2_b = 0
            else:
                part_2_b = np.sqrt(part_2_b_a - part_2_b_num / part_2_b_den)

        if flag_a:
            return part_1 + part_2_a * part_2_b  # Ascending
        else:
            return part_1 - part_2_a * part_2_b  # Descending
