from typing import final
from unittest.case import skip
import gym

from rl.tabular import TabularQLearner, TabularDynaQLearner, TabularSarsaLearner, TabularExpectedSarsaLearner
from tqdm import tqdm

import numpy as np

import unittest


class TestTabularAgents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("*********Testing Tabular Methods*********")

    def setUp(self):
        print("Unit tests")

    @skip
    def test_plot_accum_reward(self):
        import matplotlib.pyplot as plt
        env = gym.make("CliffWalking-v0")
        test_episodes = 500
        runs = 50
        rewards_q = np.zeros(test_episodes)
        rewards_sarsa = np.zeros(test_episodes)
        rewards_expected_sarsa = np.zeros(test_episodes)
        for i in range(runs):
            q_agent = TabularQLearner(env, episodes=test_episodes, gamma=1, eps=0.1, alpha=0.5, decaying_eps=False)
            sarsa_agent = TabularSarsaLearner(env, episodes=test_episodes, gamma=1, eps=0.1, alpha=0.5, decaying_eps=False)
            expected_sarsa_agent = TabularExpectedSarsaLearner(env, episodes=test_episodes, gamma=1, eps=0.1, alpha=0.5, decaying_eps=False)
            rewards_q += q_agent.learn()
            rewards_sarsa += sarsa_agent.learn()
            rewards_expected_sarsa += expected_sarsa_agent.learn()

        rewards_q /= runs
        rewards_sarsa /= runs
        rewards_expected_sarsa /= runs

        # draw reward curves
        plt.plot(rewards_sarsa, label='Sarsa')
        plt.plot(rewards_q, label='Q-Learning')
        plt.plot(rewards_expected_sarsa, label='Expected Sarsa')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.ylim([-100, 0])
        plt.legend()

        plt.savefig('figure_example_6_6.svg')
        plt.close()

    def test_tabular_ql_cliffwalking(self):
        env = gym.make("CliffWalking-v0")
        test_episodes = 500
        # agent = TabularQLearner(env, episodes=test_episodes, eps=0.1, alpha=0.5, decaying_eps=False)
        # agent = TabularDynaQLearner(env, episodes=test_episodes)
        # agent = TabularSarsaLearner(env, episodes=test_episodes)
        agent = TabularSarsaLearner(env, episodes=test_episodes, eps=0.1, alpha=0.5, decaying_eps=False)
        accum_rewards = agent.learn()
        self.assertEquals(len(accum_rewards), test_episodes)
        print(f"Accumulated rewards {accum_rewards}")

        obs = env.reset()
        final_reward = 0
        did_fall = False
        for t in range(20):
            # env.render()
            action = agent.policy(obs)
            obs, reward, done, info = env.step(action)
            if reward == -100:
                did_fall = True
            if done:
                # obs = env.reset()
                final_reward = 1
        env.close()
        self.assertEqual(1, final_reward)
        self.assertFalse(did_fall)

    def test_tabular_ql_blackjack(self):
        env = gym.make("Blackjack-v0")
        test_episodes = 200000
        agent = TabularQLearner(env, episodes=test_episodes)
        # agent = TabularDynaQLearner(env, episodes=test_episodes)
        accum_rewards = agent.learn()
        self.assertEquals(len(accum_rewards), test_episodes)
        print(f"Accumulated rewards {accum_rewards}")

        obs = env.reset()
        rewards = []
        did_win = False
        for t in range(400):
            action = agent.policy(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                rewards.append(reward)
                if reward == 1:
                    did_win = True
        env.close()
        self.assertTrue(did_win)
        print(f"Average Reward: {np.average(rewards)}")
        # self.assertEqual(1, final_reward)
