import gym
import tensorflow as tf
import numpy as np

import logging

# config logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

env_name = 'CartPole-v0'
env = gym.make(env_name)
logger.info("{} is made".format(env_name))
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
logger.info("action dimension of env is {}".format(action_dim))
logger.info("state dimension of env is {}".format(state_dim))

for i_episode in xrange(20):
    observation = env.reset()
    episode_reward = 0
    for t in xrange(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            logger.info("Episode finished after {} timesteps with total reward {}".format(t+1, episode_reward))
            break
