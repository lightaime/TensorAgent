import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
from model.reinforce_model import Model

import logging

# config logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# make environment
env_name = 'CartPole-v0'
env = gym.make(env_name)
env = wrappers.Monitor(env, env_name+"experiment-1", force=True)
logger.info("{} is made".format(env_name))
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
logger.info("action dimension of env is {}".format(action_dim))
logger.info("state dimension of env is {}".format(state_dim))

# parameters
MAX_EPISODE = 100000
MAX_STEP = 1000
DISCOUNT_FACTOR = 0.99
VERBOSE = False

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
model = Model(state_dim, action_dim)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    last_100_epi_red = []
    for i_episode in xrange(MAX_EPISODE):
        observation = env.reset()
        episode_reward = 0
        state_rollout = []
        action_rollout = []
        reward_rollout = []
        done_rollout = []
        for t in xrange(MAX_STEP):
            state_rollout.append(observation)
            # env.render()
            p=model.predict_policy([observation], sess)
            action = np.random.choice(action_dim, 1, p=p[0])
            observation, reward, done, info = env.step(action[0])
            episode_reward += reward

            action_rollout.append(action)
            if done and episode_reward != 200:
                reward = -10
            elif done and episode_reward == 200:
                reward = 10
                print("POSITIVE DONE!")
            reward_rollout.append(reward)
            done_rollout.append(done)

            if done:
                last_100_epi_red.insert(0, episode_reward)
                if len(last_100_epi_red) > 100:
                    last_100_epi_red.pop()
                logger.info("Episode {} finished after {} timesteps with total reward {}".format(i_episode, t+1, episode_reward))
                avg_reward = sum(last_100_epi_red) / float(len(last_100_epi_red))
                logger.info("Last 100 episodes average reward is {}".format(sum(last_100_epi_red) / float(len(last_100_epi_red))))
                if avg_reward >= 195.0:
                    print("PROBLEM SOLVED!")
                    exit()
                break

        w = np.array([])
        b = np.array([])
        for i, sard in enumerate(zip(state_rollout, action_rollout, reward_rollout, done_rollout)):
            s, a, r, d = sard
            _, total_loss, policy_loss, base_line_loss = model.update([s],
                            [a],
                            [[sum(DISCOUNT_FACTOR**i_ * rwd for i_, rwd in enumerate(reward_rollout[i:]))]],
                            sess)
            if VERBOSE:
                if i%10 == 0:
                    print(base_line_loss)
                    print(total_loss)
                w_p, b_p = w.copy(), b.copy()
                w, b = model.run_layer_weight()
                if i > 0:
                    print(w-w_p)
                    print(b-b_p)

        del state_rollout
        del action_rollout
        del reward_rollout
        del done_rollout
