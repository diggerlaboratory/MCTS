import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import random
import numpy as np
import torch
import utils
from PIL import Image
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
        env = gym.make(name,render_mode="rgb_array")
        env_sim = gym.make(name,render_mode="rgb_array")
    else:
        env = gym.make(name)
        env_sim = gym.make(name)
        
    env = utils.SkipFrame(env, skip=5)
    env_sim = utils.SkipFrame(env_sim, skip=5)
    for count in range(test_count):
        frameN0 = 0
        episodic_reward = 0
        seed = random.randint(0,100)
        obs,info = env.reset(seed=seed)
        obs_sim,info_sim = env_sim.reset(seed=seed)
        if save:
            frame = env.render()
            save_frame_as_image(frame=frame,file_path=f"count_{count}_frame_{frameN0}.png")
        test_root = utils.MCTSNode(obs)
        test_mcts = utils.MCTS_LunarLander(env=env, sim_env=env_sim, network=network)
        while True:
            frameN0 = frameN0+1
            test_mcts.search(test_root)
            best_action, _ = test_root.best_child(c_puct=1.0)
            next_obs, reward, done,truncated,info = env.step(best_action)
            episodic_reward = episodic_reward + reward
            if save:
                frame = env.render()
                save_frame_as_image(frame=frame,file_path=f"count_{count}_frame_{frameN0}.png")
            obs = next_obs
            test_root = utils.MCTSNode(obs)
            if done or truncated:
                break
            print(f"testing: frameNO:{frameN0} episodeNo:{count}/{test_count}")
        test_rewards.append(episodic_reward)
    print(f"test rewards: {test_rewards}")
    return np.mean(test_rewards)
            
            
            
if __name__=="__main__":
    set_seed(42)
    name = "LunarLander-v2"
    env = gym.make(name)
    env_sim = gym.make(name)
    env = utils.SkipFrame(env, skip=5)
    env_sim = utils.SkipFrame(env_sim, skip=5)
    state_dim = 8  # 상태 공간 크기
    action_dim = 4  # 행동 공간 크기
    network = utils.LunarLanderPolicyValueNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    replay_buffer = utils.ReplayBuffer(capacity=50000)
    batch_size = 256
    gamma = 0.99
    alpha = 1.0
    beta = 0.5
    print(f"PID: {os.getpid()}")
    best_reward = -100000
    for episode in range(100000000):
        print(f"episode: {episode}")
        seed = random.randint(0,1000)
        obs,info = env.reset(seed=seed)
        obs_sim,_ = env_sim.reset(seed=seed)
        
        done = False
        total_reward = 0
        root = utils.MCTSNode(obs)
        mcts = utils.MCTS_LunarLander(env=env,sim_env=env_sim, network=network)
        while True:
            mcts.search(root)
            best_action, _ = root.best_child(c_puct=1.0)
            next_obs, reward, done,truncated,info = env.step(best_action)
            _ = env_sim.step(best_action)
            replay_buffer.push(obs, best_action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            root = utils.MCTSNode(obs)
            if done or truncated:
                break
            print(f"len(replay_buffer): {len(replay_buffer)}, batch_size:{batch_size} best_action:{best_action}")
            if len(replay_buffer) >= batch_size:
                for update in range(min(512,64*2**(2*int(len(replay_buffer)/batch_size)))):
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
                test_score = doTest(name=name,network=network,test_count=5)
                print(f"episode:{episode} update:{update} test mean:{np.mean(test_score)}")
                if (test_score > best_reward):
                    best_reward = test_score
                    os.makedirs(f"./policy/{name}/{episode}/",exist_ok=True)
                    torch.save(network.state_dict(), f"./policy/{name}/{episode}/policy_{best_reward}.pth")
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


    env.close()
    env_sim.close()