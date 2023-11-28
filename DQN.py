import torch
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设备以及环境初始化
plt.ion()
env = gym.make("CartPole-v1")
gpu_id = 1
device = torch.device(f"cuda:{str(gpu_id)}" if torch.cuda.is_available() else "cpu")

# 经验回放机制





