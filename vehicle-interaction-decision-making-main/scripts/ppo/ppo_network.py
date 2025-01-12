import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Actor, self).__init__()
        self.num_agents = num_agents
        self.fc1 = nn.Linear(state_dim * num_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        # 4是k取值的个数，即k可以去0，1，2，3
        self.fc3 = nn.Linear(256, 4 ** self.num_agents)
        nn.Softmax(dim=-1)  # 输出为概率分布

    def forward(self, states):
        # 将输入的 states 转换为 float32 类型的张量
        states = torch.tensor(states, dtype=torch.float32).view(1, -1)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action_probs = self.fc3(x)
        action_probs = F.softmax(action_probs, dim=-1)  
        
        # 返回动作的概率分布
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_dim, num_agents, action_dim):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim * num_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.action_dim ** self.num_agents)

    def forward(self, states):
        states = torch.tensor(states, dtype=torch.float32).view(1, -1)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        state_values = self.fc3(x)
        return state_values

