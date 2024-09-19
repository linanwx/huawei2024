import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# 自定义环境
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        
    def reset(self):
        self.state = np.random.uniform(-10, 10, size=(1,))
        return self.state

    def step(self, action):
        self.state = self.state + action
        reward = -((self.state[0] - 1) * (self.state[0] - 2)) ** 2
        done = False
        return self.state, reward, done, {}

# 自定义策略网络
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)
        self.net_arch = [64, 64]  # 定义网络架构
        self.activation_fn = nn.ReLU
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

# 创建环境和模型
env = CustomEnv()
model = PPO(CustomPolicy, env, verbose=1)

# 训练循环
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps)

# 测试训练后的模型
test_state = np.array([0.0])
optimal_action, _ = model.predict(test_state)
print(f"For state {test_state}, the optimal action is {optimal_action}")
print(f"The optimized value is: {test_state + optimal_action}")

# 评估模型
num_episodes = 100
total_reward = 0

for _ in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    
    total_reward += episode_reward

average_reward = total_reward / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")