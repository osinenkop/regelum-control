from abc import ABC, abstractmethod
from rcognita.__utilities import rc


class AdaptationBlock(ABC):
    def __init__(self, c_hat_init, learning_rate, **kwargs):
        self.c_hat_init = c_hat_init
        self.current_c_hat = c_hat_init
        self.learning_rate = learning_rate
        self.__dict__.update(**kwargs)

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
    def __init__(self, c_hat_init, learning_rate, system=None):
        super().__init__(c_hat_init, learning_rate)
        self.m_c, self.m_p, self.g, self.l = system.pars

    def parameter_estimation_derivative(self, current_state):
        theta = current_state[0]
        x = current_state[1]
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
