import tensorflow as tf
import numpy as np

class Agent(object):
    def __init__(self, model, discout_factor, verbose=False):
        self.model = model
        self.discout_factor = discout_factor
        self.verbose = verbose
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.state_rollout = []
        self.action_rollout = []
        self.reward_rollout = []
        self.done_rollout = []

    def state_append(self, state):
        self.state_rollout.append(state)

    def action_append(self, action):
        self.action_rollout.append(action)

    def reward_append(self, reward):
        self.reward_rollout.append(reward)

    def predict_policy(self, observation):
        return self.model.predict_policy([observation], self.sess)

    def train_model(self):
        w = np.array([])
        b = np.array([])
        for i, sar in enumerate(zip(self.state_rollout, self.action_rollout, self.reward_rollout)):
            s, a, r = sar
            _, total_loss, policy_loss, base_line_loss = self.model.update([s],
                            [a],
                            [[sum(self.discout_factor**i_ * rwd for i_, rwd in enumerate(self.reward_rollout[i:]))]],
                            self.sess)
            if i%10 == 0:
                print(base_line_loss)
            if self.verbose:
                if i%10 == 0:
                    print(base_line_loss)
                    print(total_loss)
                    w_p, b_p = w.copy(), b.copy()
                    w, b = model.run_layer_weight()
                    if i > 0:
                        print(w-w_p)
                        print(b-b_p)

    def clear_rollout(self):
        del self.state_rollout[:]
        del self.action_rollout[:]
        del self.reward_rollout[:]

