import copy

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from v6_addr import v6_address, give_reward
import gym
import copy
# from gym.spaces import Discrete, Dict, Tuple

#经验重放, 有待修改
class replay_buffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, 2], dtype=np.float32) #([-1,0,1],[0-f])
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

class v6_agent:
    def __init__(
            self,
            env:gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        obs_dim = 32 + 1 + 1
        action_dim = 16 + 1 + 1

        self.env = env
        self.memory = replay_buffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update #time to hard update，not network
        self.gamma = gamma

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    #选择动作
    def select_action(self, state: list):
        #EE 17-18 times
        #e-greedy policy default: test
        if self.epsilon > np.random.random():
            if (state[2] == ""):
                selected_action = self.env.action_space.sample((np.array([0,1,1], np.int8), np.ones((16,), np.int8)))
            else:
                selected_action = self.env.action_space.sample()
        else:
            #前一位，当前位，下一位所有可能reward计算，找最大
            reward_list = []
            action_list = []
            #previous state
            if state[2] != "":
                pre_state = state
                pre_state[2] = state[2][:-1]
                pre_state[0] = int(state[2][-1], base=16)
                pre_state[1] = state[1] - 1
                reward_list.append(give_reward(pre_state))
                action_list.append((-1, pre_state[0]))
            #当前state
            reward_list.append(give_reward(state))
            action_list.append((0, state[0]))
            #next state
            new_state = state
            new_state[2] += str(hex(state[0]))[-1]
            for i in range(16):
                new_state[0] = i
                new_state[1] += 1
                reward_list.append(give_reward(state))
                action_list.append((1, i))
            pos = reward_list.index(max(reward_list))
            selected_action = action_list[pos]

        #非测试条件，创建transition，并加入state arr和action arr
        if not self.is_test:
            state_arr = np.full(34, -1)
            state_arr[0] = state[0] + 1
            state_arr[1] = state[1]
            for i,s in enumerate(state[2]):
                state_arr[i+2] = int(s, base=16) + 1
            action_arr = np.array(selected_action)

            self.transition = [state_arr, action_arr]

        return selected_action


    #采取动作
    def step(self, action)->Tuple[list, float, bool]:
        #加一步state和action的转化，转化为ndarray, 方便之后的计算
        next_state, reward, done = self.env.step(action)
        #转state、next state
        if not self.is_test:
            #next state
            next_state_arr = np.full(34, -1)
            next_state_arr[0] = next_state[0] + 1
            next_state_arr[1] = next_state[1]
            for i,s in enumerate(next_state[2]):
                next_state_arr[i+2] = int(s, base=16) + 1
            self.transition += [next_state_arr, reward, done]
        return next_state, reward, done
    #compute loss
    def compute_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 2)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        # curr_q_value = self.dqn(state).gather(1, action)
        curr_q_value = torch.tensor()
        if action[0] == -1:
            curr_q_value = self.dqn(state).gather(1,0)
        elif action[0] == 0:
            curr_q_value = self.dqn(state).gather(1,1)
        else:
            curr_q_value = self.dqn(state).gather(1,action[1] + 2)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    #update model
    def update_model(self)->torch.Tensor:
        samples = self.memory.sample_batch()
        loss = self.compute_loss(samples)
        #梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    #update target model
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def train(self, state, iteration:int, time_to_print:int):
        self.is_test = False
        scores = []
        score = 0
        losses = []
        update_cnt = 0
        epsilons = []
        state_origin = state
        for iter_times in range(1, iteration+1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                state = state_origin
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                #it's time for target network to update
                if iter_times % self.target_update == 0:
                    self._target_hard_update(self)
            if iter_times % time_to_print == 0:
                print("trained for %d times" % iter_times)

    def test(self):
        self.is_test = True
        origin_env = self.env
        state = self.env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        print("score : %d" % score)
        self.env.close()

        #reset
        self.env = origin_env
if __name__ == "__main__":
    env = v6_address()
    state_0 = env.reset()
    print(state_0)

    agent = v6_agent(env, 10, 2, target_update=5, epsilon_decay=0.2) #epsilon max 1.0 min 0.1 gamma 0.99
    action = agent.select_action(state_0)
    print(action)

    next_state, reward, done = agent.step(action)
    print(next_state, reward, done)

    agent.train(state_0, 100, 10)
    agent.test()
    print(env.observation_space)
