import json
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from utils import load_problem_data
from evaluation import evaluation_function
# from evaluation_diff import DiffInput, DiffSolution

demand, datacenters, servers, selling_prices = load_problem_data()
MAX_TIMESTEPS = 168

import numpy as np

class SlotAvailabilityManager:
    def __init__(self, datacenters):
        self.max_timesteps = MAX_TIMESTEPS
        self.dc_ids = datacenters['datacenter_id'].values
        # 创建一个3D NumPy数组：(时间步, 数据中心, 服务器类型)
        self.slots_table = np.zeros((MAX_TIMESTEPS, len(self.dc_ids)), dtype=np.int32)
        for i, dc in enumerate(self.dc_ids):
            self.slots_table[:, i] = datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0]
        
        # 存储已使用的容量，按服务器类型分类
        self.server_counts = np.zeros((MAX_TIMESTEPS, len(self.dc_ids), len(servers)), dtype=np.int32)
        self.slots_size = servers['slots_size'].to_numpy()

    def check_availability(self, start_time, end_time, dc_index, server_type, slots_needed):
        end_time = min(end_time, self.max_timesteps)
        return np.all(self.slots_table[start_time-1:end_time, dc_index] >= slots_needed * self.slots_size[server_type])
        # return self._check_availability(self.slots_table, start_time, end_time, dc_index, slots_needed)


    def get_maximum_available_slots(self, start_time, end_time, dc_index):
        end_time = min(end_time, self.max_timesteps)
        return np.min(self.slots_table[start_time-1:end_time, dc_index])
        # return self._get_maximum_available_slots(self.slots_table, start_time, end_time, dc_index)

    def update_slots(self, start_time, end_time, dc_index, server_type, change):
        end_time = min(end_time, self.max_timesteps)
        self.slots_table[start_time-1:end_time, dc_index] += change
        self.server_counts[start_time-1:end_time, dc_index, server_type] += change * self.slots_size[server_type]

    def get_total_servers_by_type(self, time_step, dc_index, server_type):
        return np.sum(self.server_counts[time_step-1, dc_index, server_type])

    def get_total_servers(self, time_step, dc_index):
        return np.sum(self.server_counts[time_step-1, dc_index, :])

class ServerStateSpace:
    def __init__(self):
        self.num_data_centers = len(datacenters)
        self.num_cpu_types = 4
        self.num_gpu_types = 3
        self.num_server_types = self.num_cpu_types + self.num_gpu_types
        
        self.data_center_id = datacenters['datacenter_id'].to_numpy()
        self.cost_of_energy = datacenters['cost_of_energy'].to_numpy()
        self.latency_sensitivity = np.array([0, 1, 2, 2])
        self.slots_capacity = datacenters['slots_capacity'].to_numpy()
        
        self.purchase_price = servers['purchase_price'].to_numpy()
        self.energy_consumption = servers['energy_consumption'].to_numpy()
        self.slots_size = servers['slots_size'].to_numpy()
        self.capacity = servers['capacity'].to_numpy()
        self.average_maintenance_fee = servers['average_maintenance_fee'].to_numpy()
        self.servers = defaultdict(list)

class ServerEnv(gym.Env):
    def __init__(self):
        super(ServerEnv, self).__init__()
        self.state_space = ServerStateSpace()
        self.current_step = 0
        self.scaler = MinMaxScaler()
        self.actions_per_combination = 30
        self.next_server_id = 0
        self.server_generation = servers['server_generation'].to_numpy()
        self.life_expectancy = servers['life_expectancy'].to_numpy()
        self.release_time = [(1, 60), (37, 96), (73, 132), (109, 168), (1, 72), (49, 120), (97, 168)]  # 每种服务器的购买时间范围
        self.solution = pd.DataFrame(columns=['time_step', 'datacenter_id',
                                              'server_generation', 'server_id', 'action'])

        # 成本计算
        self.total_cost = 0
        # self.S = DiffSolution(seed=6053)
        self.slot_manager = SlotAvailabilityManager(datacenters)

        num_base_actions = self.state_space.num_server_types * self.state_space.num_data_centers
        num_actions = num_base_actions * self.actions_per_combination
        total_slots = np.sum(self.state_space.slots_capacity) // 2

        self.action_space = spaces.MultiDiscrete([
            4,  # operation (buy, dismiss, hold, move)
            total_slots,  # server_id
            self.state_space.num_data_centers,  # target
            self.actions_per_combination
        ] * num_base_actions)

        # 简化状态空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(80,), dtype=np.float32)
    
    def get_state(self):
        # 获取所有数据中心的剩余容量
        remaining_capacity = np.array([
            self.slot_manager.get_maximum_available_slots(self.current_step, self.current_step + 1, dc)
            for dc in range(self.state_space.num_data_centers) 
        ])

        # 获取所有数据中心每种服务器类型的数量
        server_counts = np.array([
            [self.slot_manager.get_total_servers_by_type(self.current_step, dc, server_type) for server_type in range(self.state_space.num_server_types)]
            for dc in range(self.state_space.num_data_centers)
        ]).flatten()

        raw_state = np.concatenate([
            self.state_space.cost_of_energy / np.max(self.state_space.cost_of_energy),
            self.state_space.latency_sensitivity / 2,
            self.state_space.slots_capacity / np.sum(self.state_space.slots_capacity),
            self.state_space.purchase_price / np.max(self.state_space.purchase_price),
            self.state_space.energy_consumption / np.max(self.state_space.energy_consumption),
            self.state_space.slots_size / np.max(self.state_space.slots_size),
            self.state_space.capacity / np.max(self.state_space.capacity),
            self.state_space.average_maintenance_fee / np.max(self.state_space.average_maintenance_fee),
            [self.current_step / MAX_TIMESTEPS],
            remaining_capacity / np.max(self.state_space.slots_capacity),  # 归一化剩余容量
            server_counts / (np.max(self.state_space.slots_capacity) // 2)  # 归一化服务器数量
        ])
        
        normalized_state = self.scaler.fit_transform(raw_state.reshape(1, -1)).flatten()
        return normalized_state

    def step(self, action):

        # 解析动作
        # num_sub_actions = len(action) // 3
        num_base_actions = self.state_space.num_server_types * self.state_space.num_data_centers
        for i in range(num_base_actions):
            # operation, server_id, target = action[i*3:(i+1)*3]
            # data_center_idx = i // self.state_space.num_server_types
            # server_type_idx = i % self.state_space.num_server_types

            operation, server_id, target, intensity = action[i*4:(i+1)*4]
            data_center_idx = i // self.state_space.num_server_types
            server_type_idx = i % self.state_space.num_server_types

            if self.current_step == 0:
                # 在第一轮只允许购买服务器
                if operation == 0:  # 购买服务器
                    self._buy_server(data_center_idx, server_type_idx, intensity)
                else:
                    continue
            else:
              if operation == 0:  # 购买服务器
                  self._buy_server(data_center_idx, server_type_idx, intensity)
              elif operation == 1:  # 解雇服务器
                  self._dismiss_server(data_center_idx, server_type_idx, intensity)
              elif operation == 2:  # 保持不变
                  pass
              elif operation == 3:  # 移动服务器
                  self._move_server(data_center_idx, target, server_type_idx, intensity)


        # 计算奖励
        reward = self._calculate_reward()
        # reward = reward / 10000  # 缩放奖励

        

        # 检查是否结束
        new_state = self.get_state()
        self.current_step += 1
        done = self.current_step > MAX_TIMESTEPS 

        if reward == -100:
            done = True

        # 更新状态

        return new_state, reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 1
        # self.S = DiffSolution(seed=seed)
        self.state_space = ServerStateSpace()
        self.solution = pd.DataFrame(columns=['time_step', 'datacenter_id',
                                              'server_generation', 'server_id', 'action'])
        return self.get_state(), {}
    
    def _can_buy_server(self, data_center_idx, server_type_idx, num):
        server_release_time = self.release_time[server_type_idx]
        if self.current_step < server_release_time[0] or self.current_step > server_release_time[1]:
          return False
        return self.slot_manager.check_availability(self.current_step, self.current_step + 1, data_center_idx, server_type_idx, num)
        # return self.state_space.slots_capacity[data_center_idx] >= self.state_space.slots_size[server_type_idx] * num
    
    def _can_move_server(self, from_dc, to_dc, server_type_idx, num):
        is_available = self.slot_manager.check_availability(self.current_step, self.current_step + 1, to_dc, server_type_idx, num)
        total_servers = self.slot_manager.get_total_servers_by_type(self.current_step, from_dc, server_type_idx)
        return is_available and total_servers >= num
        # return self.state_space.slots_capacity[to_dc] >= self.state_space.slots_size[server_type_idx] * num

    def _buy_server(self, data_center_idx, server_type_idx, intensity):

        num_server = max(intensity, 1)
        if self._can_buy_server(data_center_idx, server_type_idx, num_server):
            self.slot_manager.update_slots(self.current_step,
                                           self.current_step + self.life_expectancy[server_type_idx],
                                           data_center_idx,
                                           server_type_idx,
                                           num_server
                                           )
            
            # 添加新服务器到追踪系统
            for i in range(num_server):
              new_server = (self.next_server_id, server_type_idx, self.current_step, self.current_step)
              self.state_space.servers[data_center_idx].append(new_server)
              self.next_server_id += 1
              self.solution.loc[len(self.solution)] = [
                  self.current_step,
                  self.state_space.data_center_id[data_center_idx],
                  self.server_generation[new_server[1]],
                  new_server[0],
                  "buy"
              ]
    
    def _dismiss_server(self, data_center_idx, server_type_idx, intensity):
        num_servers = max(1, int(intensity)) 
        total_servers = self.slot_manager.get_total_servers(self.current_step, data_center_idx)
        if total_servers >= intensity:
            # self.state_space.server_counts[data_center_idx, server_type_idx] -= num_servers
            # self._update_slots(data_center_idx, server_type_idx, num_servers)
            
            # 从追踪系统中移除最旧的服务器
            servers = self.state_space.servers[data_center_idx]
            count = 0 
            life_expectancy = self.life_expectancy[server_type_idx]
            for i, server in enumerate(servers):
                if server[1] == server_type_idx:
                    server_age = self.current_step - server[2]  # 计算服务器年龄
                    if server_age > life_expectancy:
                        continue
                    count += 1
                    removed_server = servers.pop(i)
                    self.slot_manager.update_slots(removed_server[2], removed_server[2] + life_expectancy, data_center_idx, server_type_idx, -1)
                    # self.total_cost += self._calculate_dismissal_cost(removed_server)
                    self.solution.loc[len(self.solution)] = [
                      self.current_step,
                      self.state_space.data_center_id[data_center_idx],
                      self.server_generation[removed_server[1]],
                      removed_server[0],
                      "dismiss"
                    ]
                    if count == num_servers:
                        break

    def _move_server(self, from_dc, to_dc, server_type_idx, intensity):
        num_servers = max(1, int(intensity))
        slot_size = self.state_space.slots_size[server_type_idx]
        if self._can_move_server(from_dc, to_dc, server_type_idx, num_servers) and \
            from_dc != to_dc:

            # 更新服务器位置
            count = 0
            life_expectancy = self.life_expectancy[server_type_idx]
            servers = self.state_space.servers[from_dc]
            for i, server in enumerate(servers):
                if server[1] == server_type_idx:
                    server_age = self.current_step - server[2]  # 计算服务器年龄
                    if server_age > life_expectancy:
                        continue
                    count += 1
                    moved_server = servers.pop(i)
                    updated_server = (moved_server[0], moved_server[1], moved_server[2], self.current_step)
                    left_lifespan = life_expectancy - server_age
                    self.slot_manager.update_slots(self.current_step, self.current_step + left_lifespan, to_dc, server_type_idx, 1)
                    self.slot_manager.update_slots(moved_server[2], moved_server[2] + life_expectancy, from_dc, server_type_idx, -1)

                    self.state_space.servers[to_dc].append(updated_server)
                    self.solution.loc[len(self.solution)] = [
                      self.current_step,
                      self.state_space.data_center_id[to_dc],
                      self.server_generation[updated_server[1]],
                      updated_server[0],
                      "move"
                    ]
                    if count == num_servers:
                      break

    def _calculate_reward(self):
        reward = evaluation_function(self.solution,
                    demand,
                    datacenters,
                    servers,
                    selling_prices,
                    seed=6053)
        if reward is None:
            return -100
        # input:DiffInput = DiffInput(is_new=True, step=1, diff_solution=self.solution)
        # try:
        #     reward = self.S.SA_evaluation_function(input)
        # except:
        #     solution = self.solution
        #     data = solution.to_dict('records')
        #     with open('error_example.json', 'w') as f:
        #         json.dump(data, f, indent=2)
        #     return -100
        print("step: {}, reward: {}".format(self.current_step, reward))
        return reward / 1000000
    def get_solution(self):
        return self.solution

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * n_input, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations.unsqueeze(1))

def make_env():
    def _init():
        env = ServerEnv()
        return env
    return _init

def save_solution(env, filename):
    solution = env.get_solution()
    data = solution.to_dict('records')
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def train():
    num_cpu = 2  # 设置为您的CPU核心数
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])
    # env = DummyVecEnv([make_env()])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    learning_rate = 1e-3
    end_learning_rate = 1e-5
    learning_rate_schedule = get_linear_fn(learning_rate, end_learning_rate, 1000000)

    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=learning_rate_schedule,
                n_steps=1024 // num_cpu,
                batch_size=64 // num_cpu,
                n_epochs=10,
                ent_coef=0.01,
                clip_range=0.2,
                policy_kwargs=policy_kwargs,
                # use_sde=True,
                # sde_sample_freq=4,
                )

    # eval_env = DummyVecEnv([make_env()])
    eval_env = SubprocVecEnv([make_env() for i in range(num_cpu)])
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000,
                                 deterministic=True, render=False)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoints/',
                                             name_prefix='ppo_server_model')

    model.learn(total_timesteps=int(1e4), callback=[checkpoint_callback, eval_callback])

    model.save("ppo_server_model")


def test():
    env = DummyVecEnv([make_env()])
    model = PPO.load("ppo_server_model", env=env)

    for i in range(20):
        done = False
        score = 0
        obs = env.reset()
        while not done:
          action, _states = model.predict(obs, deterministic=True)
          obs, rewards, dones, info = env.step(action)
          score = rewards
        save_solution(env, f'train_solution_{i}.json')

if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()