import gym
from gym import wrappers
import numpy as np
import random
from model.reinforce_model import Model
from agent.reinforce import Agent

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
DISCOUNT_FACTOR = 0.97
ENTROPY_BETA = 1e-3
LEARNING_RATE = 0.001
VERBOSE = False

model = Model(state_dim, action_dim, entropy_beta=ENTROPY_BETA, learning_rate=LEARNING_RATE)
agent = Agent(model, DISCOUNT_FACTOR, VERBOSE)
last_100_epi_red = []
for i_episode in xrange(MAX_EPISODE):
    observation = env.reset()
    episode_reward = 0
    for t in xrange(MAX_STEP):
        agent.state_append(observation)
        # env.render()
        p = agent.predict_policy(observation)
        action = np.random.choice(action_dim, 1, p=p[0])
        observation, reward, done, info = env.step(action[0])
        episode_reward += reward

        if done and episode_reward != 200:
            reward = -10
        elif done and episode_reward == 200:
            reward = 10
            print("positive done!")
        agent.action_append(action)
        agent.reward_append(reward)

        if done:
            last_100_epi_red.insert(0, episode_reward)
            if len(last_100_epi_red) > 100:
                last_100_epi_red.pop()
            logger.info("episode {} finished after {} timesteps with total reward {}".format(i_episode, t+1, episode_reward))
            avg_reward = sum(last_100_epi_red) / float(len(last_100_epi_red))
            logger.info("last 100 episodes average reward is {}".format(sum(last_100_epi_red) / float(len(last_100_epi_red))))
            if avg_reward >= 195.0:
                print("problem solved!")
                exit()
            break
    agent.train_model()
    agent.clear_rollout()
