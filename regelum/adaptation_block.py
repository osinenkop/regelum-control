"""Contains adaptation blocks."""

from abc import ABC, abstractmethod
from regelum.__utilities import rc


class AdaptationBlock(ABC):
    """An interface for controller parameters adaptation."""

    def __init__(self, c_hat_init, learning_rate):
        """Initialize an instance of AdaptationBlock.

        :param c_hat_init:  initial value of adapted parameter
        :param learning_rate: rate of adaptation
        """
        self.c_hat_init = c_hat_init
        self.current_c_hat = c_hat_init
        self.learning_rate = learning_rate

    @abstractmethod
    def parameter_estimation_derivative(self, current_state):
        pass

    def euler_step(self, current_state):
        self.current_c_hat += self.learning_rate * self.parameter_estimation_derivative(
            current_state
        )

        return self.current_c_hat

    def get_parameter_estimation(self, current_state):
        return self.euler_step(current_state)


class AdaptationBlockCartpole(AdaptationBlock):
    """An adaptation block for Cartpole."""

    def __init__(self, c_hat_init, learning_rate, system=None):
        """Initialize an instance of AdaptationBlockCartpole.

        :param c_hat_init:  initial value of adapted parameter
        :param learning_rate: rate of adaptation
        :param system: an instance of Cartpole system to be adapted to
        """
        super().__init__(c_hat_init, learning_rate)
        self.m_c, self.m_p, self.g, self.l = (
            system.parameters["m_c"],
            system.parameters["m_p"],
            system.parameters["g"],
            system.parameters["l"],
        )

    def parameter_estimation_derivative(self, current_state):
        theta = current_state[0]
        theta_dot = current_state[2]
        x_dot = current_state[3]

        return (
            self.learning_rate
            * self.m_p
            * self.l
            * theta_dot
            * rc.cos(theta)
            * 1
            / (self.m_c + self.m_p * rc.sin(theta) ** 2)
            * x_dot**2
            * rc.sign(x_dot)
        )
