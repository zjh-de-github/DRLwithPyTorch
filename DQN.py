# from:https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio
from collections import namedtuple, deque
from itertools import count

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设备以及环境初始化
plt.ion()
env = gym.make("CartPole-v1", render_mode = "rgb_array")
gpu_id = 1
device = torch.device(f"cuda:{str(gpu_id)}" if torch.cuda.is_available() else "cpu")

# 经验回放机制
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplyMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen = capacity) # 经验池最大容量,使用了deque双端队列
    def push(self, *args):
        # 保存一个存储元组
        self.memory.append(Transition(*args))   
    def sample(self, bach_size):
        # 从经验池中随机取样
        return random.sample(self.memory, bach_size)
    def __len__(self):
        return len(self.memory)

# 接下来就是DQN的算法流程
# 阅读原文 https://www.nature.com/articles/nature14236?source=post_page---------------------------
# 或者知乎教程 https://zhuanlan.zhihu.com/p/110620815

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # 三个全连接层
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, input):
        # 输入是观测, 输出是动作
        input = F.relu(self.layer1(input))
        input = F.relu(self.layer2(input))
        return self.layer3(input)

# 接下来是超参的设置
# 动作选取action_select
# 可视化画图plot_durations

BATH_SIZE = 128
GAMMA = 0.90 # 对未来奖励的可视程度
EPS_START = 0.8 # 开始时选择动作的随机性
EPS_END = 0.05 # 快结束时选择动作的随机性
EPS_DECAY = 1000 # 指数
TAU = 0.005 # 学习率
LR = 1e-4 # 优化器学习率

n_actions = env.action_space.n # 从gym中获取动作数量
state, info = env.reset()
n_observations = len(state) # 状态张量

# 将目标网络和策略网络部署在设备上
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # 初始化目标网络

optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)
memory = ReplyMemory(10000)

steps_done = 0

def action_select(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long, requires_grad=False)
    
episode_durations = []

def plot_durations(show_result = False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def display_frames_as_gif(frames, id):
    print(frames[0])
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 1)
    anim.save(f'./GIF/CartPole-v1/episodes_{id}.gif', writer='ffmpeg', fps = 30)

def optimize_model():
    if len(memory) < BATH_SIZE:
        return 
    transitions = memory.sample(BATH_SIZE)
    # print(transitions)
    bath = Transition(*zip(*transitions))
    # 遍历bath中的每一个下一状态是否为最终态，并且生成一个张量
    non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, bath.next_state)), device = device, dtype=torch.bool) 
    # cat在默认维度 0维 上拼接list 并且返回tensor
    non_final_next_states = torch.cat([s for s in bath.next_state if s is not None])
    state_batch = torch.cat(bath.state)
    action_batch = torch.cat(bath.action)
    reward_batch = torch.cat(bath.reward)
    # Q网络的输入是state_batch，输出就是state对应的Q(s,a)，再用gather将两者对应起来
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATH_SIZE, device = device)
    # torch.no_grad() 就是不进行网络更新反向传播，仅仅是作为输入输出进行使用
    with torch.no_grad():
        # 用下一个状态中最大的Q值来估计这个状态的V值
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values
    excepted_state_action_values = GAMMA * next_state_values + reward_batch # 根据下一步的状态计算出期望的Q值
    criterion = nn.SmoothL1Loss()
    # unsqueeze(1)将excepted_state_action_values变成了一个(BATH_SIZE,1)维度的张量，与state_action_values相匹配
    loss = criterion(state_action_values, excepted_state_action_values.unsqueeze(1))
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episodes in range(num_episodes):
    frames = []
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = action_select(state)
        # action.item()用于获取aciton中的标量，只有在tensor中只有一个量的时候才能调用
        frames.append(env.render())
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated # 到达终态或者异常停止

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            # 可视化
            gif_path = f'./GIF/CartPole-v1/episodes_{i_episodes}.gif'
            # display_frames_as_gif(frames=frames, id = i_episodes)
            imageio.mimsave(gif_path, frames)
            print(f"episode_{i_episodes}/{num_episodes} have done!")
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()






