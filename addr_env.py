import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import gym
from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
from copy import deepcopy
# from typing import Dict, List, Tuple



class v6_address(gym.Env):
    def __init__(self):
        self.length = 32
        self.start = [np.random.choice(0xf), 0, ""] #当前位，position，prefix
        self.curr_state = None
        self.action_space = Tuple((Discrete(3, start=-1), Discrete(0xf))) #第一位：方向：1-》，0：stay，-1《-；第二位：变成什么
        # self.observation_space = Tuple((Discrete(0xf), Discrete(32)))

    def step(self, action):
        next_state = deepcopy(self.curr_state)
        #向右移动
        if action[0] == 1:
            next_state[2] += str(hex(next_state[0])[-1])
            next_state[0] = action[1]
            next_state[1] += 1
        elif action[0] == 0:
            pass
        elif action[0] == -1:
            next_state[2] = next_state[2][:-1] #prefix
            next_state[0] = action[1] #核心值
            next_state[1] -= 1 #pos
        else:
            raise Exception("invalid action")
        self.curr_state = next_state

        if self.curr_state[1] == 31:
            done = True
        else :
            done = False
        reward = give_reward(self.curr_state)
        info = {}

        return self.curr_state, reward, done, info
    def render(self): #可视化，先不搞了
        pass
    def reset(self):
        self.curr_state = self.start
        return self.curr_state

def give_reward(state:list)->int:
    reward = EE(state[2] + str(hex(state[0])[-1]))
    return reward

def EE(sth):
    return np.random.random()

if __name__ == "__main__":
    env = v6_address()
    env.curr_state = env.start
    env.step((1,0))
    print(env.curr_state)
    print(env.step((1,0xf)))