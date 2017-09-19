import numpy as np

class Agent(object):
    def __init__(self, model, replay_buffer, exploration_noise, discout_factor, verbose=False):
        self.model = model
        self.replay_buffer = replay_buffer
        self.exploration_noise = exploration_noise
        self.discout_factor = discout_factor
        self.verbose = verbose

    def predict_action(self, observation):
        return self.model.predict_action(observation)

    def select_action(self, observation, p=None):
        pred_action = self.predict_action(observation)
        noise = self.exploration_noise.return_noise()
        if p is not None:
            return pred_action * p + noise * (1 - p)
        else:
            return pred_action + noise

    def store_transition(self, transition):
        self.replay_buffer.store_transition(transition)

    def init_process(self):
        self.exploration_noise.init_process()

    def get_transition_batch(self):
        batch = self.replay_buffer.get_batch()
        transpose_batch = list(zip(*batch))
        s_batch = np.vstack(transpose_batch[0])
        a_batch = np.vstack(transpose_batch[1])
        r_batch = np.vstack(transpose_batch[2])
        next_s_batch = np.vstack(transpose_batch[3])
        done_batch = np.vstack(transpose_batch[4])
        return s_batch, a_batch, r_batch, next_s_batch, done_batch

    def preprocess_batch(self, s_batch, a_batch, r_batch, next_s_batch, done_batch):
        target_actor_net_pred_action = self.model.actor.predict_action_target_net(next_s_batch)
        target_critic_net_pred_q = self.model.critic.predict_q_target_net(next_s_batch, target_actor_net_pred_action)
        y_batch = r_batch + self.discout_factor * target_critic_net_pred_q * (1 - done_batch)
        return s_batch, a_batch, y_batch

    def train_model(self):
        s_batch, a_batch, r_batch, next_s_batch, done_batch = self.get_transition_batch()
        self.model.update(*self.preprocess_batch(s_batch, a_batch, r_batch, next_s_batch, done_batch))


