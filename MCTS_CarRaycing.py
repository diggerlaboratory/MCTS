import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from PIL import Image
import utils
import gymnasium
from gymnasium.spaces import Discrete
class DiscreteCarRacingWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(5)  # 5개의 이산 행동
        # 이산 행동을 연속형 행동으로 매핑
        self._action_map = {
            0: np.array([0.0, 0.0, 0.0]),  # 정지
            1: np.array([-1.0, 0.0, 0.0]), # 좌회전
            2: np.array([1.0, 0.0, 0.0]),  # 우회전
            3: np.array([0.0, 1.0, 0.0]),  # 가속
            4: np.array([0.0, 0.0, 1.0]),  # 브레이크
        }

    def action(self, act):
        return self._action_map[act]
    
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
    

def save_frame_as_image(frame, file_path):
    """
    Gym 환경의 프레임(numpy 배열)을 이미지로 저장합니다.
    """
    # Numpy 배열을 PIL 이미지로 변환
    image = Image.fromarray(frame)
    # 파일로 저장
    image.save(file_path)
    
def doTest(name,network,test_count,save=False):
    test_rewards = []
    if save:
        env = gymnasium.make(name,render_mode="rgb_array")
        env_sim = gymnasium.make(name,render_mode="rgb_array")
    else:
        env = gymnasium.make(name)
        env_sim = gymnasium.make(name)
        
    env = DiscreteCarRacingWrapper(env)
    env = utils.SkipFrame(env, skip=5)

    env_sim = DiscreteCarRacingWrapper(env_sim)
    env_sim = utils.SkipFrame(env_sim, skip=5)
    for count in range(test_count):
        frameN0 = 0
        episodic_reward = 0
        seed = random.randint(0,10)
        print(f"seed: {seed}")
        obs,info = env.reset(seed=seed)
        obs_sim,info_sim = env_sim.reset(seed=seed)
        if save:
            frame = env.render()
            os.makedirs(f"./renderImage/{name}/count_{count}/",exist_ok=True)
            save_frame_as_image(frame=frame,file_path=f"./renderImage/{name}/count_{count}/frame_{frameN0}.png")
        episodic_reward = 0
        obs,info = env.reset()
        obs = np.transpose(obs, (2, 0, 1)) / 255.0
        test_root = utils.MCTSNode(obs)
        test_mcts = utils.MCTS(env=env,sim_env=env_sim,network=network)
        while True:
            frameN0 = frameN0+1
            test_mcts.search(test_root)
            best_action, _ = test_root.best_child(c_puct=1.0)
            next_obs, reward, done,truncated,info = env.step(best_action)
            if save:
                frame = env.render()
                save_frame_as_image(frame=frame,file_path=f"./renderImage/{name}/count_{count}/frame_{frameN0}.png")
            next_obs = np.transpose(next_obs, (2, 0, 1)) / 255.0
            obs = next_obs
            test_root = utils.MCTSNode(obs)
            episodic_reward = episodic_reward + reward
            print(f"testing: frameNO:{frameN0} episodeNo:{count}/{test_count}")
            print(f"reward:{reward}, done:{done}, truncated:{truncated},info: {info}")
            if done or truncated:
                break
        test_rewards.append(episodic_reward)
    return np.mean(test_rewards)

if __name__=="__main__":
    name = "CarRacing-v2"
    env = gymnasium.make(name)
    env = DiscreteCarRacingWrapper(env)
    env = utils.SkipFrame(env, skip=5)
    env_sim = gymnasium.make(name)
    env_sim = DiscreteCarRacingWrapper(env_sim)
    env_sim = utils.SkipFrame(env_sim, skip=5)
    set_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    obs_shape = env.observation_space.shape
    network = utils.ResNetPolicyValueNetwork(obs_shape, env.action_space.n).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    replay_buffer = utils.ReplayBuffer(capacity=50000)
    batch_size = 128
    gamma = 0.99
    alpha = 1.0
    beta = 0.5
    best_reward = -100000
    print(f"PID: {os.getpid()}")
    for episode in range(100000000):
        print(f"episode: {episode}")
        seed = random.randint(0,1000)
        obs,info = env.reset(seed=seed)
        _ = env_sim.reset(seed=seed)
        obs = np.transpose(obs, (2, 0, 1)) / 255.0
        total_reward = 0
        root = utils.MCTSNode(obs)
        mcts = utils.MCTS(env=env,sim_env=env_sim, network=network)
        while True:
            mcts.search(root)
            best_action, _ = root.best_child(c_puct=1.0)
            next_obs, reward, done,truncated,info = env.step(best_action)
            _ = env_sim.step(best_action)
            next_obs = np.transpose(next_obs, (2, 0, 1)) / 255.0
            replay_buffer.push(obs, best_action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            root = utils.MCTSNode(obs)
            if done or truncated:
                break
            print(f"len(replay_buffer): {len(replay_buffer)}, batch_size:{batch_size} best_action:{best_action}")
            if len(replay_buffer) >= batch_size:
                for update in range(32):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                    actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

                    with torch.no_grad():
                        _, next_values = network(next_states_tensor)
                        targets = rewards_tensor + gamma * next_values * (1 - dones_tensor)

                    policy_logits, predicted_values = network(states_tensor)
                    policy_loss = nn.CrossEntropyLoss()(policy_logits, actions_tensor)
                    value_loss = nn.MSELoss()(predicted_values, targets)
                    loss = alpha * policy_loss + beta * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print(f"len(replay_buffer): {len(replay_buffer)}, batch_size:{batch_size} update count:{update}/{min(512,2**(2*int(len(replay_buffer)/batch_size)))}")
                test_score = doTest(name=name,network=network,test_count=2,save=False)
                print(f"episode:{episode} update:{update} test mean:{np.mean(test_score)}")
                if (test_score > best_reward):
                    best_reward = test_score
                    os.makedirs(f"./policy/{name}/{episode}/",exist_ok=True)
                    torch.save(network.state_dict(), f"./policy/{name}/{episode}/policy_{best_reward}.pth")

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
