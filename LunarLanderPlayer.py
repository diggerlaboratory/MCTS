import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from gym.wrappers import TimeLimit
import random
import numpy as np
import torch
import utils
from MCTS_LunarLander import doTest
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def set_seed(seed):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
name = "LunarLander-v2"
env = gym.make(name)
state_dim = 8  # 상태 공간 크기
action_dim = 4  # 행동 공간 크기
network = utils.LunarLanderPolicyValueNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(network.parameters(), lr=1e-4)
replay_buffer = utils.ReplayBuffer(capacity=50000)
batch_size = 256
gamma = 0.99
alpha = 1.0
beta = 0.5
network.load_state_dict(torch.load("/home/user0005/0001_MCTS_GYM/policy/LunarLander-v2/3/policy_-189.1758676536979.pth"))
doTest(name=name,network=network,test_count=1)