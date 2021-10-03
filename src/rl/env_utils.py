import gym
from gym import Env
from gym.envs.toy_text.cliffwalking import UP, DOWN, LEFT, RIGHT, CliffWalkingEnv
from rl.agent import RLAgent
from IPython.display import HTML, display


def print_cliffwalking_policy(agent: RLAgent, env: CliffWalkingEnv):
    GOAL = (env.shape[0]-1, env.shape[1]-1)
    optimal_policy = []
    for i in range(env.shape[0]):
        optimal_policy.append([])
        for j in range(env.shape[1]):
            if (i, j) == GOAL:
                optimal_policy[-1].append("G")
                continue
            bestAction = agent.policy((i, j))
            if bestAction == UP:
                optimal_policy[-1].append('U')
            elif bestAction == DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)


TERMINAL = '&#x25CE;'


def display_cliffwalking_policy(agent: RLAgent, env: CliffWalkingEnv):
    GOAL = (env.shape[0]-1, env.shape[1]-1)
    optimal_policy = []
    for i in range(env.shape[0]):
        optimal_policy.append([])
        for j in range(env.shape[1]):
            if (i, j) == GOAL:
                optimal_policy[-1].append(TERMINAL)
                continue
            bestAction = agent.policy((i, j))
            if bestAction == UP:
                optimal_policy[-1].append('&uarr;')
            elif bestAction == DOWN:
                optimal_policy[-1].append('&darr;')
            elif bestAction == LEFT:
                optimal_policy[-1].append('&larr;')
            elif bestAction == RIGHT:
                optimal_policy[-1].append('&rarr;')
    display(HTML(
        '<table style="font-size:300%;border: thick solid;"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in optimal_policy))))
