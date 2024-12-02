import torchvision.models as models
from collections import deque
import torch.nn as nn
import numpy as np
import random
import torch
import gymnasium as gym
import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 리플레이 버퍼 정의
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# MCTS 노드 정의
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def get_value(self):
        return self.value_sum / (1 + self.visit_count)

    def add_child(self, action, next_state):
        child = MCTSNode(next_state, parent=self)
        self.children[action] = child
        return child

    def best_child(self, c_puct):
        best_score = -float('inf')
        best_action = None
        for action, child in self.children.items():
            ucb = child.get_value() + c_puct * np.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
        return best_action, self.children[best_action]

# MCTS 알고리즘
class MCTS:
    def __init__(self, env,sim_env, network, c_puct=1.0, simulations=128, gamma=0.99):
        self.env = env
        self.sim_env = sim_env
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations
        self.gamma = gamma

    def _simulate(self, node):
        """
        시뮬레이션 단계: 환경 없이 정책 네트워크를 사용하여
        현재 상태의 가치만 반환.
        """
        # 현재 상태 가져오기
        state = node.state

        # 정책 네트워크를 통해 행동 확률과 상태 가치 계산
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.network(state_tensor)  # 상태 가치 출력

        # 상태 가치를 반환
        return value.item()


    def search(self, root):
        """MCTS 탐색 과정."""
        for _ in tqdm.trange(self.simulations):
            node = root
            while node.children:
                _, node = node.best_child(self.c_puct)
            
            # Expansion
            state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, _ = self.network(state_tensor)  # 정책 네트워크 출력
            action_probs = policy.cpu().numpy().squeeze()
            untried_actions = [a for a in range(len(action_probs)) if a not in node.children]
            untried_probs = action_probs[untried_actions]
            untried_probs /= untried_probs.sum()  # 확률 정규화
            untried_action = np.random.choice(untried_actions, p=untried_probs)
            
            # next_state, _, _, _,_ = self.env.step(untried_action)
            next_state, _, _, _,_ = self.sim_env.step(untried_action)
            
            next_state = self._preprocess_state(next_state)
            node = node.add_child(untried_action, next_state)

            value = self._simulate(node)

            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent

    def _preprocess_state(self, state):
        return np.transpose(state, (2, 0, 1)) / 255.0

class ResNetPolicyValueNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(ResNetPolicyValueNetwork, self).__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim

        # ResNet-101 기반 특징 추출기
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()  # Fully Connected 레이어 제거
        self.feature_extractor = resnet

        # 정책 헤드 (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet-101 출력 크기 2048
            nn.ReLU(),
            nn.Linear(512, action_dim),  # 행동 공간 크기
            nn.Softmax(dim=-1),  # 확률 분포로 변환
        )

        # 상태 가치 헤드 (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # 스칼라 값 출력
        )

    def forward(self, x):
        # 특징 추출
        features = self.feature_extractor(x)

        # 정책 및 상태 가치 계산
        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value
    
class LunarLanderPolicyValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LunarLanderPolicyValueNetwork, self).__init__()
        
        # 공통 네트워크 (Shared Layers)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),  # 입력: 상태 크기
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 정책 헤드 (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Linear(16, action_dim),  # 출력: 행동 공간 크기
            nn.Softmax(dim=-1)           # 행동 확률로 변환
        )
        
        # 가치 헤드 (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(16, 1)            # 출력: 스칼라 가치 값
        )

    def forward(self, state):
        # 공통 네트워크
        shared_features = self.shared(state)
        
        # 정책 및 가치 계산
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        return policy, value

# MCTS 알고리즘
class MCTS_LunarLander:
    def __init__(self, env,sim_env, network, c_puct=1.0, simulations=128, gamma=0.99):
        self.env = env
        self.sim_env = sim_env
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations
        self.gamma = gamma
    
    def _simulate(self, node):
        """
        시뮬레이션 단계: 환경 없이 정책 네트워크를 사용하여
        현재 상태의 가치만 반환.
        """
        # 현재 상태 가져오기
        state = node.state
        # 정책 네트워크를 통해 행동 확률과 상태 가치 계산
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.network(state_tensor)  # 상태 가치 출력

        # 상태 가치를 반환
        return value.item()


    def search(self, root):
        """MCTS 탐색 과정."""
        for _ in tqdm.trange(self.simulations):
            node = root
            while node.children:
                _, node = node.best_child(self.c_puct)
            # Expansion
            state_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, _ = self.network(state_tensor)  # 정책 네트워크 출력
            action_probs = policy.cpu().numpy().squeeze()
            untried_actions = [a for a in range(len(action_probs)) if a not in node.children]
            untried_probs = action_probs[untried_actions]
            untried_probs /= untried_probs.sum()  # 확률 정규화
            untried_action = np.random.choice(untried_actions, p=untried_probs)
            # next_state, _, _, _,_ = self.env.step(untried_action)
            next_state, _, _, _,_  = self.sim_env.step(untried_action)
            node = node.add_child(untried_action, next_state)
            value = self._simulate(node)

            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info