import torch
import torch.nn as nn
import torch.optim as optim
import random
from gym.wrappers import TimeLimit
import random
import numpy as np
import torch
import utils
import os
import gymnasium as gym
import gymnasium
from MCTS_CarRaycing import doTest,DiscreteCarRacingWrapper
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
if __name__=="__main__":
    name = "CarRacing-v2"
    env = gymnasium.make(name)
    env = utils.SkipFrame(env,1)
    env = DiscreteCarRacingWrapper(env)
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    obs_shape = env.observation_space.shape
    network = utils.ResNetPolicyValueNetwork(obs_shape, env.action_space.n).to(device)
    # network.load_state_dict(torch.load("/home/user0005/0001_MCTS_GYM/policy/CarRacing-v2/1/policy_-58.277235104547046.pth"))
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    replay_buffer = utils.ReplayBuffer(capacity=50000)
    batch_size = 128
    gamma = 0.99
    alpha = 1.0
    beta = 0.5
    print(f"PID: {os.getpid()}")
    doTest(name=name,network=network,test_count=2,save=True)