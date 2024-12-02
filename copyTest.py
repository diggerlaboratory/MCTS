import gym
import copy
import numpy as np

# 기존 환경 생성 및 step 진행
env = gym.make("CarRacing-v2")
env_copy = gym.make("CarRacing-v2")

obs = env.reset()
copy_obs = env_copy.reset()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done,truncated, info = env.step(action)
    _,_,_,_,_ = env_copy.step(action)
    if done:
        break

action = env.action_space.sample()
obs_original, reward_original, done_original,tr, info_original = env.step(action)
obs_copy, reward_copy, done_copy,tr_copy,info_copy = env_copy.step(action)

print("원본 환경 결과:", obs_original, reward_original, done_original, info_original,tr)
print("복사된 환경 결과:", obs_copy, reward_copy, done_copy,tr_copy,info_copy)
