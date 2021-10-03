from typing import Any, Collection, List, NoReturn, Sequence, overload
from random import Random
from abc import abstractmethod
from math import log2, exp
from gym.core import Env
from tqdm import tqdm

import numpy as np

# These definitions are for typing (we may want to move this elsewhere)
State = Any
Action = int


def softmax(values: List[float]) -> List[float]:
    """Computes softmax probabilities for an array of values
    TODO We should probably use numpy arrays here
    Args:
        values (np.array): Input values for which to compute softmax

    Returns:
        np.array: softmax probabilities
    """
    return [(exp(q))/sum([exp(_q) for _q in values]) for q in values]


class RLAgent:
    """
    This is a base class used as parent class for any
    RL agent. This is currently not much in use, but is
    recommended as development goes on.
    """
    def __init__(self,
                 env: Env,
                 episodes: int = 100,
                 decaying_eps: bool = True,
                 eps: float = 0.9,
                 alpha: float = 0.01,
                 decay: float = 0.00005,
                 gamma: float = 0.99,
                 rand: Random = Random()):
        self.env = env
        self.episodes = episodes
        self.decaying_eps = decaying_eps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        self._random = rand

    @abstractmethod
    def agent_start(self, state: State) -> Action:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (Action): the first action the agent takes.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, state: State) -> Action:
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            (Action): The action the agent is taking.
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float) -> NoReturn:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    @abstractmethod
    def policy(self, state: State) -> Action:
        """The action for the specified state under the currently learned policy
           (unlike agent_step, this does not update the policy using state as a sample.
           Args:
                state (State): the state observation from the environment
           Returns:
                (Action): The action prescribed for that state
        """
        pass

    @abstractmethod
    def epsilon_greedy_policy(self, state: State) -> Action:
        """Returns the epsilon-greedy policy

        Args:
            state (State): The state for which to return the epsilon greedy policy

        Returns:
            (Action): The action to be taken
        """
        pass

    @abstractmethod
    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        pass

    @abstractmethod
    def learn(self, max_tsteps: int = float("inf")) -> Sequence[float]:
        """Learns a policy within the environment set at initialization.

        Args:
            max_tsteps[int]: The maximum time steps per episode
                             (useful in non-terminal environments)

        Returns:
            Sequence[float]: The sequence of accumulated rewards in each episode
        """
        done_times = 0

        accum_rewards = []
        max_r = float("-inf")
        print(f'Training agent: {self.__class__.__name__} in environment {self.env.__class__.__name__}')
        tq = tqdm(range(self.episodes), postfix=f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
        for n in tq:
            self.step = n
            episode_r = 0
            state = self.env.reset()
            action = self.agent_start(state)
            done = False
            tstep = 0
            total_reward = 0
            while tstep < max_tsteps and not done:
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    done_times += 1

                action = self.agent_step(reward, obs)
                tstep += 1
                episode_r += reward
            if done:  # One last update at the terminal state
                self.agent_end(reward)

            accum_rewards.append(total_reward)

            if episode_r > max_r:
                max_r = episode_r
                # print("New all time high reward:", episode_r)
                tq.set_postfix_str(f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
            if (n + 1) % 100 == 0:
                tq.set_postfix_str(f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
            if (n + 1) % 1000 == 0:
                tq.set_postfix_str(f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
                # print(f'Episode {n+1} finished. Timestep: {tstep}. Number of states: {len(self.q_table.keys())}. Reached the goal {done_times} times during this interval. Eps = {self.c_eps}')
                done_times = 0

        return accum_rewards

    def __getitem__(self, state: State) -> Any:
        """[summary]

        Args:
            state (Any): The state for which we want to get the policy

        Raises:
            InvalidAction: [description]

        Returns:
            Any: [description]
        """""
        return self.policy(state)
