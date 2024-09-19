# ppo_sa_env.py

import os
from colorama import Fore, Style
import gym
from gym import spaces
import numpy as np
from threading import Event

from stable_baselines3 import PPO
import torch

from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo
from real_diff_SA_basic import OperationContext, SlotAvailabilityManager

def create_flatten_array_exclusive(servers_df, w):
    # 初始化二维数组列表
    result = []
    
    # 遍历 servers_df 的每一行
    for _, row in servers_df.iterrows():
        # 创建一个全为0的数组，长度为w
        array = np.zeros(w)
        # 从 release_start - 1 到 release_end - 2 设置为1 (不包含 release_end - 1)
        array[row['release_start'] - 1 : row['release_end'] - 1] = 1
        result.append(array)
    
    # 将二维数组合并为一维数组
    flattened_array = np.concatenate(result).flatten()
    
    return flattened_array

class PPO_SA_Env(gym.Env):
    """
    Custom Environment for integrating PPO with SA BuyServerOperation
    """
    metadata = {'render.modes': ['human']}
    MODEL_SAVE_PATH = 'ppo_sa_model.pth'
    SAVE_INTERVAL = 100  # Save model every 100 steps, can be adjusted
    step_counter = 0

    def _print(self, *args, color=None, **kwargs):
        if self.context.verbose:
            # 设置颜色
            if color:
                print(f"{color}{' '.join(map(str, args))}{Style.RESET_ALL}", **kwargs)
            else:
                print(*args, **kwargs)

    def __init__(self, sa_solution:DiffSolution, context:OperationContext, slot_manager: SlotAvailabilityManager, servers_df, id_gen, model=None):
        super(PPO_SA_Env, self).__init__()
        self.sa_solution = sa_solution  # SA's solution object
        self.slot_manager = slot_manager
        self.servers_df = servers_df
        self.id_gen = id_gen
        self.context = context

        # Define action and observation space
        # Actions: data_center, server_type, start_time, end_time, slots_needed
        self.action_space = spaces.MultiDiscrete([
            len(self.slot_manager.total_slots),  # data_center
            len(self.servers_df),                # server_type
            168,                                 # start_time
            168,                                 # end_time
            10                                  # quantity
        ])

        # Observations: We'll use the capacity matrix, demand matrix, satisfaction matrix, current score
        # Flatten the matrices for simplicity
        self.observation_space = spaces.Dict({
            'capacity_matrix': spaces.Box(low=0, high=np.inf, shape=(168 * 3 * 7,), dtype=np.float32),
            'demand_matrix': spaces.Box(low=0, high=np.inf, shape=(168 * 3 * 7,), dtype=np.float32),
            'satisfaction_matrix': spaces.Box(low=0, high=np.inf, shape=(168 * 3 * 7,), dtype=np.float32),
            'current_score': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'buy_time_info': spaces.Box(low=0, high=1, shape=(168 * 7,), dtype=np.float32)
        })

        # Synchronization events for communication between SA and PPO
        self.signal1 = Event()  # PPO signals that step is done
        self.signal2 = Event()  # SA waits for PPO to be ready

        self.current_observation = None
        self.model = model

    def save_model(self):
        if self.model is not None:
            print(f"Saving model to {self.MODEL_SAVE_PATH}")
            self.model.save(self.MODEL_SAVE_PATH)  # Save the model to the path

    def get_state(self):
        self.current_observation = {
            'capacity_matrix': self.sa_solution.capacity_matrix.copy().flatten(),
            'demand_matrix': self.sa_solution.demand_matrix.copy().flatten(),
            'satisfaction_matrix': self.sa_solution.satisfaction_matrix.copy().flatten(),
            'current_score': np.array([self.context.sa_status.current_score], dtype=np.float32),
            'buy_time_info': create_flatten_array_exclusive(self.servers_df, 168)
        }
        return self.current_observation

    def reset(self):
        # Wait for signal2 from SA
        self.signal2.wait()
        self.signal2.clear()
        
        return self.get_state()
    
    def decode_action(self, action):
        data_center_idx = action[0]
        server_type_idx = action[1]
        start_time = action[2]
        end_time = action[3]
        quantity = action[4]
        return data_center_idx, server_type_idx, start_time, end_time, quantity

    def step(self, action):
        data_center_idx, server_type_idx, start_time, end_time, quantity = self.decode_action(action)
        self._print(f'PPO action info: data_center_idx: {data_center_idx}, server_type_idx: {server_type_idx}, start_time: {start_time}, end_time: {end_time}, quantity: {quantity}')

        # 检查约束条件
        success, constraint_violation = self.check_constraints(
            data_center_idx, server_type_idx, start_time, end_time, quantity
        )

        if not success:
            self._print(f'Constraint violation: {constraint_violation}', color=Fore.RED)
            # Apply penalty
            reward = -10000000000  # Fixed penalty value
            new_score = 0
        else:
            # Apply the action to SA and get new score
            self._print('Applying action to SA...', color=Fore.GREEN)
            new_score = self.apply_action_to_sa(action)
            
            reward = new_score - self.context.sa_status.current_score
            self._print(f'Reward: {reward}')
            self._print(f'New score: {new_score}')
            self._print(f'Current score: {self.context.sa_status.current_score}')

        # 保存本轮结果
        self.info = {'constraint_violation': constraint_violation, 'score': new_score}
        # Signal SA that step is done
        self.signal1.set()

        # Wait for signal2 from SA for the next state
        self.signal2.wait()
        self.signal2.clear()

        self.step_counter += 1

        # 每隔设定步数保存一次模型
        if self.step_counter % self.SAVE_INTERVAL == 0:
            self.save_model()

        done = False  # SA controls when to terminate, so we always return False

        return self.get_state(), reward, done, self.info

    def check_constraints(self, data_center_idx, server_type_idx, start_time, end_time, quantity):

        if start_time >= end_time:
            return False, 'Start time is greater than or equal to end time'
        if end_time - start_time > self.servers_df.iloc[server_type_idx]['life_expectancy']:
            return False, 'Life expectancy is less than the time range'

        data_center = list(self.slot_manager.total_slots.keys())[data_center_idx]
        server_info = self.servers_df.iloc[server_type_idx]
        slots_size = server_info['slots_size']

        total_slots_needed = quantity * slots_size

        # Check slot availability
        if not self.slot_manager.check_availability(start_time, end_time, data_center, total_slots_needed):
            return False, 'Not enough slots available'

        # Check purchasing time is within valid range
        if not (server_info['release_start'] <= start_time + 1 <= server_info['release_end']):
            return False, 'Start time is not within the server release time range'

        return True, None

    def apply_action_to_sa(self, action):
        # self._print(f'{self.context.solution.server_map}') # 检查一下map是不是有异常，应该是空的
        data_center_idx, server_type_idx, start_time, end_time, quantity = self.decode_action(action)

        data_center = list(self.slot_manager.total_slots.keys())[data_center_idx]
        selected_server = self.servers_df.iloc[server_type_idx]
        server_generation = selected_server['server_generation']
        slots_size = selected_server['slots_size']

        # Create ServerInfo and apply change to SA solution
        server_id = self.id_gen.next_id()
        buy_and_move_info = [ServerMoveInfo(time_step=start_time, target_datacenter=data_center)]
        server_info = ServerInfo(
            server_id=server_id,
            server_generation=server_generation,
            quantity=quantity,
            dismiss_time=end_time,
            buy_and_move_info=buy_and_move_info
        )
        self.sa_solution.apply_server_change(server_info)
        self.slot_manager.push_slot_update(start_time, end_time, data_center, quantity * slots_size, 'buy')

        # Calculate new score
        new_score = self.sa_solution.diff_evaluation()
        return new_score

    def render(self, mode='human'):
        pass

    def close(self):
        pass


    