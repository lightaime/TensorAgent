'''
refer to openai
https://github.com/rll/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
'''

import numpy as np

class OU_Process(object):
    def __init__(self, action_dim, theta=0.15, mu=0, sigma=0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.current_x = None

        self.init_process()

    def init_process(self):
        self.current_x = np.ones(self.action_dim) * self.mu

    def update_process(self):
        dx = self.theta * (self.mu - self.current_x) + self.sigma * np.random.randn(self.action_dim)
        self.current_x = self.current_x + dx

    def return_noise(self):
        self.update_process()
        return self.current_x

if __name__ == "__main__":
    ou = OU_Process(3, theta=0.15, mu=0, sigma=0.2)
    states = []
    for i in range(10000):
        states.append(ou.return_noise()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()

