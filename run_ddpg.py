from model.ddpg_model import Model
from agent.ddpg import Agent
from mechanism.replay_buffer import Replay_Buffer
from mechanism.ou_process import OU_Process
from gym import wrappers
import gym
import numpy as np

ENV_NAME = 'Pendulum-v0'
EPISODES = 100000
MAX_EXPLORE_EPS = 100
TEST_EPS = 1
BATCH_SIZE = 64
BUFFER_SIZE = 1e6
WARM_UP_MEN = 5 * BATCH_SIZE
DISCOUNT_FACTOR = 0.99
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
TAU = 0.001

def main():
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, ENV_NAME+"experiment-1", force=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = Model(state_dim,
                  action_dim,
                  actor_learning_rate=ACTOR_LEARNING_RATE,
                  critic_learning_rate=CRITIC_LEARNING_RATE,
                  tau=TAU)
    replay_buffer = Replay_Buffer(buffer_size=int(BUFFER_SIZE) ,batch_size=BATCH_SIZE)
    exploration_noise = OU_Process(action_dim)
    agent = Agent(model, replay_buffer, exploration_noise, discout_factor=DISCOUNT_FACTOR)

    action_mean = 0
    i = 0
    for episode in range(EPISODES):
        state = env.reset()
        agent.init_process()
        # Training:
        for step in range(env.spec.timestep_limit):
            # env.render()
            state = np.reshape(state, (1, -1))
            if episode < MAX_EXPLORE_EPS:
                p = episode / MAX_EXPLORE_EPS
                action = np.clip(agent.select_action(state, p), -1.0, 1.0)
            else:
                action = agent.predict_action(state)
            action_ = action * 2
            next_state, reward, done, _ = env.step(action_)
            next_state = np.reshape(next_state, (1, -1))
            agent.store_transition([state, action, reward, next_state, done])
            if agent.replay_buffer.memory_state()["current_size"] > WARM_UP_MEN:
                agent.train_model()
            else:
                i += 1
                action_mean = action_mean + (action - action_mean) / i
                print("running action mean: {}".format(action_mean))
            state = next_state
            if done:
                break

        # Testing:
        if episode % 2 == 0 and episode > 10:
            total_reward = 0
            for i in range(TEST_EPS):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    # env.render()
                    state = np.reshape(state, (1, 3))
                    action = agent.predict_action(state)
                    action_ = action * 2
                    state, reward, done, _ = env.step(action_)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward/TEST_EPS
            print("episode: {}, Evaluation Average Reward: {}".format(episode, avg_reward))

if __name__ == '__main__':
    main()
