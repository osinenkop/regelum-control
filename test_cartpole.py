import numpy as np
import gym


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


env = gym.make("CartPole-v1")
desired_state = np.array([0, 0, 0, 0])
desired_mask = np.array([1, 1, 1, 1])

P, I, D = (
    np.array([1 / 150, 1 / 950, 0.1, 0.01]),
    np.array([0.0005, 0.001, 0.01, 0.0001]),
    np.array([0.2, 0.0001, 0.5, 0.005]),
)

for i_episode in range(1):
    state = env.reset()
    integral = 0
    derivative = 0
    prev_error = 0
    for t in range(5000):
        error = state[0] - desired_state

        integral += error
        derivative = error - prev_error
        prev_error = error

        pid = np.dot(P * error + I * integral + D * derivative, desired_mask)
        action = sigmoid(pid)
        action = np.round(action).astype(np.int32)

        state, reward, done, info = env.step(action)
        env.render()
        # if done:
        #     print("Episode finished after {} timesteps".format(t + 1))
        #     break
env.close()
