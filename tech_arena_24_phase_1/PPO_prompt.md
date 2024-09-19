### 任务描述

将 **PPO（Proximal Policy Optimization）** 算法集成到模拟退火算法中，在 real_diff_SA.py 中实现**购买（buy）**操作的策略优化。具体要求如下：

使用 Stable-Baselines3 实现一个PPO环境
    初始化时，获得SA的指针
    动作 为 数据中心 服务器类型 购买起始时间点 购买结束时间 购买服务器所占用的插槽数量
    状态 为 sa的solution 的一些成员变量， 包括 capacity_matrix demand_matrix satisfaction_matrix 等 以及当前的分数
    奖励机制 如果 违反了约束 则固定一个惩罚值 如果没有违反约束 则调用评估函数 获得新评分，新评分和 当前分数 的差值作为奖励分数
    违反约束包括 起始时间大于结束时间 购买的数量超过插槽数量 购买的时间不在能购买的时间范围内 等， 参考原来的购买代码中关于约束检查的部分
    reset流程：
        - 等待信号2，也就是等待SA准备发起临域变换
        - 返回新状态

    step流程：
        - 约束检查
        - 计算奖励
        - 拷贝 状态 避免SA主程序更改
        - 发送信号1，通知临域变换完成，写入当前score
        - 等待信号2，等待下一次发起临域变换，也就是等待SA环境完成状态修改
        - 接收临域变换的新状态
        - 返回新状态 奖励制 done 总是为false

实现一个 继承 NeighborhoodOperation 的 PPO_BuyServerOperation 类
    发送信号2 通知环境，环境在稍后开始执行 step
    等待信号1
    读取新的分数，接受约束检查是否成功

修改主函数逻辑，集成SA和PPO learn，使得两个线程互相通信

### 原始代码(部分实现限于篇幅省略)

```python
# real_diff_evaluation.py
import copy
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List
from evaluation import get_actual_demand

TIME_STEPS = 168  # 时间下标从0到167
FAILURE_RATE = 0.0726  # 故障率

# 时延敏感性映射
LATENCY_SENSITIVITY_MAP = {'low': 0, 'medium': 1, 'high': 2}

# 服务器代次映射
SERVER_GENERATION_MAP = {
    'CPU.S1': 0,
    'CPU.S2': 1,
    'CPU.S3': 2,
    'CPU.S4': 3,
    'GPU.S1': 4,
    'GPU.S2': 5,
    'GPU.S3': 6
}

def load_global_data():
    """
    Load server, datacenter, and selling price data into global dictionaries.
    This function should be called before any ServerInfo instances are created.
    """
    global server_info_dict, datacenter_info_dict, selling_price_dict

    # Load server data
    servers_df = pd.read_csv('./data/servers.csv')
    servers_df['server_generation'] = servers_df['server_generation'].astype(str)
    server_info_dict = servers_df.set_index('server_generation').to_dict('index')

    # Load datacenter data
    datacenters_df = pd.read_csv('./data/datacenters.csv')
    datacenters_df['datacenter_id'] = datacenters_df['datacenter_id'].astype(str)
    datacenter_info_dict = datacenters_df.set_index('datacenter_id').to_dict('index')

    # Load selling prices
    selling_prices_df = pd.read_csv('./data/selling_prices.csv')
    selling_prices_df['server_generation'] = selling_prices_df['server_generation'].astype(str)
    selling_prices_df['latency_sensitivity'] = selling_prices_df['latency_sensitivity'].astype(str)
    selling_price_dict = selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()

# Load the data at the module level
load_global_data()

@dataclass
class ServerMoveInfo:
    """
    表示服务器移动信息
    """
    # 需要传入的属性
    time_step: int  # 时间下标，从0开始
    target_datacenter: str
    # 其他联查属性，不需要传入
    cost_of_energy: float = None          # 能耗成本
    latency_sensitivity: int = None       # 时延敏感性
    selling_price: float = None           # 售价

@dataclass
class ServerInfo:
    """
    服务器信息
    """
    # 需要传入的属性
    server_id: str                       # 服务器ID
    dismiss_time: int                    # 下线时间，最小为1，最大为TIME_STEPS
    buy_and_move_info: List[ServerMoveInfo]      # 服务器购买和迁移信息列表
    quantity: int                        # 服务器数量
    server_generation: str               # 服务器代次
    # 其他联查属性，不需要传入
    purchase_price: float = None         # 购买价格
    slots_size: int = None               # 服务器槽位大小
    energy_consumption: int = None       # 能耗
    capacity: int = None                 # 容量
    life_expectancy: int = None          # 最大寿命
    cost_of_moving: float = None         # 迁移成本
    average_maintenance_fee: float = None  # 平均维护费用

    def __post_init__(self):
        # 填充服务器信息
        server_info = server_info_dict.get(self.server_generation)
        if server_info is None:
            raise ValueError(f"服务器代次 {self.server_generation} 未找到。")
        self.purchase_price = server_info['purchase_price']
        self.slots_size = server_info['slots_size']
        self.energy_consumption = server_info['energy_consumption']
        self.capacity = server_info['capacity']
        self.life_expectancy = server_info['life_expectancy']
        self.cost_of_moving = server_info['cost_of_moving']
        self.average_maintenance_fee = server_info['average_maintenance_fee']
        self.init_buy_and_move_info()

    def init_buy_and_move_info(self):
        # 处理购买和移动信息
        for move in self.buy_and_move_info:
          self.process_move_info(move)

        # dismiss 时间不得超过最大寿命
        if self.buy_and_move_info:
            first_move_time = self.buy_and_move_info[0].time_step
            max_dismiss_time = first_move_time + self.life_expectancy
            if max_dismiss_time < self.dismiss_time:
                self.dismiss_time = max_dismiss_time

    def process_move_info(self, move: ServerMoveInfo):
        datacenter_id = move.target_datacenter
        datacenter_info = datacenter_info_dict.get(datacenter_id)
        if datacenter_info is None:
            raise ValueError(f"数据中心 {datacenter_id} 未找到。")

        move.cost_of_energy = datacenter_info['cost_of_energy']
        latency_sensitivity_str = datacenter_info['latency_sensitivity']
        if latency_sensitivity_str not in LATENCY_SENSITIVITY_MAP:
            raise ValueError(f"未知的时延敏感性: {latency_sensitivity_str}")
        move.latency_sensitivity = LATENCY_SENSITIVITY_MAP[latency_sensitivity_str]

        # 查找售价
        key = (self.server_generation, latency_sensitivity_str)
        selling_price = selling_price_dict.get(key)
        if selling_price is None:
            raise ValueError(f"售价未找到，服务器代次：{self.server_generation}，时延敏感性：{latency_sensitivity_str}")
        move.selling_price = selling_price

@dataclass
class DiffBlackboard:
    lifespan_percentage_sum: np.ndarray
    lifespan: np.ndarray
    fleetsize: np.ndarray
    capacity_matrix: np.ndarray
    cost: np.ndarray
    satisfaction_matrix: np.ndarray
    changed_capacity_indices: set
    utilization_matrix: np.ndarray
    average_utilization: np.ndarray
    average_lifespan: np.ndarray
    changed_time_steps: set = field(default_factory=set)

class DiffSolution:
    def __init__(self, seed, verbose=False):
        # 加载数据
        self._load_data()

        self._init_price_matrix()
        # 初始化一个空解
        self.server_map: Dict[str, ServerInfo] = {}
        # 初始化每一个时间步骤的服务器容量
        self.capacity_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的需求矩阵
        self.demand_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的满足矩阵
        self.satisfaction_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的服务器寿命
        self.lifespan = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的寿命百分比总和
        self.lifespan_percentage_sum = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的服务器数量
        self.fleetsize = np.zeros(TIME_STEPS, dtype=float)
        # 初始化平均寿命百分比
        self.average_lifespan = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的成本
        self.cost = np.zeros(TIME_STEPS, dtype=float)
        # 差分黑板，用于暂存变动
        self.__blackboard: DiffBlackboard = None
        # 当前的差分信息
        self.__diff_info: ServerInfo = None
        # 初始化利用率矩阵
        self.utilization_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化平均利用率
        self.average_utilization = np.zeros(TIME_STEPS, dtype=float)

        self._load_demand_data('./data/demand.csv', seed)
        self.__verbose = verbose

        self.cost_details = {t: {'purchase_cost': 0, 'energy_cost': 0, 'maintenance_cost': 0} for t in range(TIME_STEPS)}
        self.capacity_combinations = {t: {} for t in range(TIME_STEPS)} # 日志用

    def get_server_copy(self, server_id:str):
        server_info = self.server_map.get(server_id)
        if server_info is None:
            return None
        return copy.deepcopy(server_info)
        
    def __print(self, message):
        if self.__verbose:
            print(message)

    def _init_price_matrix(self):
        num_latencies = len(LATENCY_SENSITIVITY_MAP)
        num_servers = len(SERVER_GENERATION_MAP)
        self.price_matrix = np.zeros((num_latencies, num_servers), dtype=float)
        for latency_key, latency_idx in LATENCY_SENSITIVITY_MAP.items():
            for server_key, server_idx in SERVER_GENERATION_MAP.items():
                price_key = (server_key, latency_key)
                selling_price = self.selling_price_dict.get(price_key, 0)
                self.price_matrix[latency_idx, server_idx] = selling_price

    def _load_demand_data(self, file_path: str, seed):
        """
        加载需求数据，并填充到 demand_matrix 中。
        """
        # 读取 CSV 文件
        demand_df = pd.read_csv(file_path)
        np.random.seed(seed)
        demand_df = get_actual_demand(demand_df)

        # 假设 get_actual_demand 返回的 DataFrame 包含以下列：
        # 'time_step', 'server_generation', 'low', 'medium', 'high'
        # 我们需要将其转换为适合填充 demand_matrix 的格式

        # 将 'server_generation' 映射到索引
        demand_df['server_idx'] = demand_df['server_generation'].map(SERVER_GENERATION_MAP)
        # 将 'time_step' 映射到索引（假设时间步从1开始）
        demand_df['time_idx'] = demand_df['time_step'] - 1  # 时间步索引从0开始

        # 遍历所有的 'latency_sensitivity'，即 'low'，'medium'，'high'
        for latency in ['low', 'medium', 'high']:
            # 获取对应的 'latency_sensitivity' 索引
            latency_idx = LATENCY_SENSITIVITY_MAP[latency]

            # 提取需求数据，确保没有缺失值
            demands = demand_df[['time_idx', 'server_idx', latency]].dropna()

            # 将数据转换为 numpy 数组
            time_indices = demands['time_idx'].astype(int).values
            server_indices = demands['server_idx'].astype(int).values
            demand_values = demands[latency].astype(float).values

            # 使用索引将需求值填充到 demand_matrix 中
            self.demand_matrix[time_indices, latency_idx, server_indices] = demand_values

    def _load_data(self):
        """
        加载售价数据
        """
        # 加载售价数据
        self.selling_prices_df = pd.read_csv('./data/selling_prices.csv')
        self.selling_prices_df['server_generation'] = self.selling_prices_df['server_generation'].astype(str)
        self.selling_prices_df['latency_sensitivity'] = self.selling_prices_df['latency_sensitivity'].astype(str)
        # 建立 (server_generation, latency_sensitivity) 的售价字典
        self.selling_price_dict = self.selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()

    def discard_server_changes(self):
        """
        放弃当前对服务器的变动操作。
        """
        self.__diff_info = None
        self.__blackboard = None

    def commit_server_changes(self):
        """
        将当前对服务器的变动操作，应用到解中。
        """
        # 将 blackboard 中的数据正式写入到解的内部状态
        self.lifespan = self.__blackboard.lifespan
        self.lifespan_percentage_sum = self.__blackboard.lifespan_percentage_sum
        self.fleetsize = self.__blackboard.fleetsize
        self.capacity_matrix = self.__blackboard.capacity_matrix
        self.cost = self.__blackboard.cost
        self.utilization_matrix = self.__blackboard.utilization_matrix
        self.average_utilization = self.__blackboard.average_utilization
        self.average_lifespan = self.__blackboard.average_lifespan
        self.satisfaction_matrix = self.__blackboard.satisfaction_matrix
        # 更新 server_map
        diff_info = self.__diff_info
        if diff_info.quantity == 0:
            # 如果数量为0，表示删除该服务器
            if diff_info.server_id in self.server_map:
                del self.server_map[diff_info.server_id]
        else:
            # 添加或更新服务器
            self.server_map[diff_info.server_id] = diff_info
        # 清空当前的差分信息和黑板
        self.__diff_info = None
        self.__blackboard = None

    def apply_server_change(self, diff_info: ServerInfo):
        if diff_info.dismiss_time > diff_info.buy_and_move_info[0].time_step + diff_info.life_expectancy:
            raise ValueError("Dismiss time cannot exceed maximum lifespan.")
        if self.__blackboard is None:
            # Initialize blackboard
            self.__blackboard = DiffBlackboard(
                lifespan=self.lifespan.copy(),
                lifespan_percentage_sum=self.lifespan_percentage_sum.copy(),
                fleetsize=self.fleetsize.copy(),
                capacity_matrix=self.capacity_matrix.copy(),
                cost=self.cost.copy(),
                satisfaction_matrix=self.satisfaction_matrix.copy(),
                changed_capacity_indices=set(),
                utilization_matrix=self.utilization_matrix.copy(),
                average_utilization=self.average_utilization.copy(),
                average_lifespan=self.average_lifespan.copy(),
                changed_time_steps=set()
            )
        self.__diff_info = diff_info
        blackboard = self.__blackboard
        original_server_info = self.server_map.get(diff_info.server_id)

        # Reverse changes for original server if it exists
        if original_server_info is not None:
            self._apply_change(blackboard, original_server_info, sign=-1)

        # Apply new server changes
        if diff_info.quantity > 0:
            self._apply_change(blackboard, diff_info, sign=1)

    def _apply_change(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        应用指定的差分信息，更新黑板数据，不修改解的内部状态。
        """
        # 调整时间步骤寿命和寿命百分比总和
        self._update_lifespan(blackboard, diff_info, sign=sign)
        # 调整时间步骤服务器数量
        self._update_fleet_size(blackboard, diff_info, sign=sign)
        # 调整容量矩阵
        self._update_capacity(blackboard, diff_info, sign=sign)
        # 购买成本
        self._update_buy_cost(blackboard, diff_info, sign=sign)
        # 移动成本
        self._update_moving_cost(blackboard, diff_info, sign=sign)
        # 能耗成本
        self._update_energy_cost(blackboard, diff_info, sign=sign)
        # 维护成本
        self._update_maintenance_cost(blackboard, diff_info, sign=sign)

    def _update_lifespan(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新服务器寿命和寿命百分比总和。
        """
        pass

    def _update_fleet_size(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新服务器数量。
        """
        pass

    def _update_buy_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新购买成本。
        """
        pass

    def _update_moving_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新迁移成本。
        """
        pass

    def _update_energy_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新能耗成本。
        """
        pass

    def _update_maintenance_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新维护成本。
        """
        pass

    def _recalculate_satisfaction(self, blackboard: DiffBlackboard):
        """
        根据容量变化的区域，重新计算需求满足数组，并更新利用率矩阵。
        """
        pass

    def _adjust_capacity_by_failure_rate_approx(self, x, avg_failure_rate=FAILURE_RATE):
        return x * (1 - avg_failure_rate)

    def _update_capacity(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新容量矩阵，并记录容量变化的索引。
        """
        pass

    def _calculate_average_utilization(self, blackboard: DiffBlackboard):
        pass

    def _calculate_average_lifespan(self, blackboard: DiffBlackboard):
        pass

    def _calculate_revenue(self, blackboard: DiffBlackboard):
        revenue = np.sum(blackboard.satisfaction_matrix * self.price_matrix, axis=(1, 2))
        return revenue  # 返回每个时间步的收入

    def _calculate_profit(self, blackboard: DiffBlackboard):
        """
        计算每个时间步的利润。
        利润 = 收入 - 成本
        """
        # 先计算收入
        revenue = self._calculate_revenue(blackboard)

        # 利润 = 收入 - 成本
        profit = revenue - blackboard.cost

        return profit  # 返回每个时间步的利润

    def _final_calculation(self, blackboard: DiffBlackboard):
        """
        根据 blackboard 中的数据，进行复杂的重算工作，计算评估值。
        最终评估值 = 每步 (平均利用率 * 平均寿命 * 利润) 的乘积和
        """
        # 计算每个时间步的平均利用率
        average_utilization = self._calculate_average_utilization(blackboard)

        # 计算每个时间步的平均寿命百分比
        average_lifespan = self._calculate_average_lifespan(blackboard)

        # 计算每个时间步的利润
        profit = self._calculate_profit(blackboard)

        # 计算每个时间步的乘积： 平均利用率 * 平均寿命 * 利润
        stepwise_product = average_utilization * average_lifespan * profit

        # 计算所有时间步乘积的总和
        evaluation_result = np.sum(stepwise_product)

        if self.__verbose:
            self.__print("\n\n\nEvaluation Result:")
            for t in range(TIME_STEPS):
                self.__print({
                    'time-step': t + 1,
                    'U': round(average_utilization[t], 2),
                    'L': round(average_lifespan[t], 2),
                    'P': round(profit[t], 2),
                    'Size': int(blackboard.fleetsize[t]),
                    '购买费用': round(self.cost_details[t]['purchase_cost'], 2),
                    '能耗费用': round(self.cost_details[t]['energy_cost'], 2),
                    '维护费用': round(self.cost_details[t]['maintenance_cost'], 2),
                    '总费用': round(blackboard.cost[t], 2),
                    '总收入': round(self._calculate_revenue(blackboard)[t], 2),
                })
                self.__print(f"Time Step {t + 1} - Capacity Combinations: {self.capacity_combinations[t]}")
        return evaluation_result
    
    def diff_evaluation(self):
        """
        差分评估函数
        """
        if self.__blackboard is None:
            print("No blackboard data. Apply server changes before evaluation.")
            return 0.0
        # Recalculate satisfaction
        self._recalculate_satisfaction(self.__blackboard)

        evaluation_result = self._final_calculation(self.__blackboard)
        return evaluation_result

def export_solution_to_json(server_map: Dict[str, ServerInfo], file_path: str):
    pass

def update_best_solution(old_best: Dict[str, 'ServerInfo'], current: Dict[str, 'ServerInfo']) -> Dict[str, 'ServerInfo']:
    """
    更新旧的最优解，使其与当前解保持同步，仅对有变化的部分进行拷贝。

    参数:
        old_best (Dict[str, ServerInfo]): 旧的最优解的 server_map。
        current (Dict[str, ServerInfo]): 当前解的 server_map。

    返回:
        Dict[str, ServerInfo]: 更新后的旧的最优解的 server_map。
    """
    pass

```

```python
# real_diff_SA.py
import ast
import os
import time
import copy
import random
from typing import Dict
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from colorama import Fore, Style

# Import the new DiffSolution and related classes
from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo, export_solution_to_json, update_best_solution
from idgen import ThreadSafeIDGenerator

TIME_STEPS = 168
DEBUG = True

# Automatically create output directory
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class SlotAvailabilityManager:
    def __init__(self, datacenters, time_steps, verbose=False):
        self.verbose = verbose
        self.datacenter_slots = {}
        self.time_steps = time_steps
        self.pending_updates = []  # 存储待处理的插槽更新

        for _, row in datacenters.iterrows():
            dc_id = row['datacenter_id']
            slots_capacity = row['slots_capacity']
            # 初始化每个数据中心的插槽为一个 numpy 数组，表示各个时间步的插槽容量
            self.datacenter_slots[dc_id] = np.full(time_steps, slots_capacity, dtype=int)
        self.total_slots = {dc: datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0] for dc in datacenters['datacenter_id']}

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def check_availability(self, start_time, end_time, data_center, slots_needed):
        # 前闭后开: [start_time, end_time)
        slots = self.datacenter_slots[data_center][start_time:end_time]
        return np.all(slots >= slots_needed)

    def get_maximum_available_slots(self, start_time, end_time, data_center):
        # 前闭后开: [start_time, end_time)
        slots = self.datacenter_slots[data_center][start_time:end_time]
        return np.min(slots)

    def update_slots(self, start_time, end_time, data_center, slots_needed, operation='buy'):
        # 前闭后开: [start_time, end_time)
        if operation == 'buy':
            self.datacenter_slots[data_center][start_time:end_time] -= slots_needed
        elif operation == 'cancel':
            self.datacenter_slots[data_center][start_time:end_time] += slots_needed
        self._print(f"Updated slots in datacenter {data_center} from time {start_time} to {end_time} with operation '{operation}' and slots_needed {slots_needed}")

    def push_slot_update(self, start_time, end_time, data_center, slots_needed, operation):
        self.pending_updates.append((start_time, end_time, data_center, slots_needed, operation))
        self._print(f"Pushed slot update: operation={operation}, time={start_time}-{end_time}, datacenter={data_center}, slots_needed={slots_needed}")
    
    def apply_pending_updates(self):
        self._print(f"Applying {len(self.pending_updates)} pending slot updates.")
        for update in self.pending_updates:
            start_time, end_time, data_center, slots_needed, operation = update
            self.update_slots(start_time, end_time, data_center, slots_needed, operation)
        self.pending_updates.clear()
        self._print("All pending updates applied and cleared.")
    
    def clear_pending_updates(self):
        self._print(f"Clearing {len(self.pending_updates)} pending slot updates without applying them.")
        self.pending_updates.clear()

    def can_accommodate_servers(self, servers: Dict[str, ServerInfo]) -> bool:
        "实现略"
        self._print("所有服务器都可以被独立容纳。")
        return True
    
    def find_time_step(self, data_center, slots_needed, time_range_start, time_range_end, sign = 1):
        ret = None
        self._print(f'Finding time step in data center {data_center} for slots_needed {slots_needed} in time range {time_range_start} to {time_range_end}')
        # self._print(f'{self.datacenter_slots[data_center][time_range_start:time_range_end + 1]}')
        if sign == 1:
            for time_step in range(time_range_start, time_range_end + 1):
                if self.datacenter_slots[data_center][time_step] >= slots_needed:
                    ret = time_step
                else:
                    break
        if sign == -1:
            for time_step in reversed(range(time_range_start, time_range_end + 1)):
                if self.datacenter_slots[data_center][time_step] >= slots_needed:
                    ret = time_step
                else:
                    break
        return ret

    
@dataclass
class OperationContext:
    slot_manager: SlotAvailabilityManager
    servers_df: pd.DataFrame
    id_gen: ThreadSafeIDGenerator
    solution: DiffSolution
    verbose: bool = False
    sa_status: 'SA_status'

class NeighborhoodOperation(ABC):
    def __init__(self, context: OperationContext):
        self.context = context

    def _print(self, *args, color=None, **kwargs):
        if self.context.verbose:
            # 设置颜色
            if color:
                print(f"{color}{' '.join(map(str, args))}{Style.RESET_ALL}", **kwargs)
            else:
                print(*args, **kwargs)

    @abstractmethod
    def execute_and_evaluate(self):
        pass

class SA_NeighborhoodOperation(NeighborhoodOperation):
    def execute_and_evaluate(self):
        success = self.execute()
        if success:
            score = self.context.solution.diff_evaluation()
            return success, score
        else:
            return success, 0
        
    @abstractmethod
    def execute(self):
        pass

class BuyServerOperation(SA_NeighborhoodOperation):
    MAX_PURCHASE_RATIO = 0.12
    def execute(self):
        time_step = random.randint(0, TIME_STEPS - 1)  # 随机选择一个时间步
        data_center = random.choice(list(self.context.slot_manager.total_slots.keys()))

        # 获取该时间步可用的服务器 注意表中的时间戳是从1开始的
        available_servers = self.context.servers_df[
            (self.context.servers_df['release_start'] <= time_step + 1) &
            (self.context.servers_df['release_end'] >= time_step + 1)
        ]

        if available_servers.empty:
            self._print(f"No available servers at time step {time_step}")
            return False

        selected_server = available_servers.sample(n=1).iloc[0]
        server_generation = selected_server['server_generation']
        life_expectancy = selected_server['life_expectancy']
        slots_size = selected_server['slots_size']

        # 确保 dismiss_life 不会让时间超过 TIME_STEPS
        max_life = TIME_STEPS - time_step
        dismiss_life = random.randint(1, min(life_expectancy, max_life))
        dismiss_time = time_step + dismiss_life

        # 获取最大可用插槽数量
        max_available_slots = self.context.slot_manager.get_maximum_available_slots(time_step, dismiss_time, data_center)

        if max_available_slots < slots_size:
            self._print(f"Not enough slots {max_available_slots} in datacenter {data_center} for server {server_generation} to buy at time {time_step}", color=Fore.YELLOW)
            return False

        # 计算最大购买数量，基于 MAX_PURCHASE_RATIO
        max_quantity_based_on_ratio = int((max_available_slots // slots_size) * self.MAX_PURCHASE_RATIO)
        if max_available_slots // slots_size >= 1 and max_quantity_based_on_ratio == 0:
            max_quantity_based_on_ratio = 1 # 确保至少购买1个

        purchase_quantity = random.randint(1, max_quantity_based_on_ratio)
        # 计算所需插槽
        total_slots_needed = purchase_quantity * slots_size
        self._print(f"Max available slots: {max_available_slots}, Max purchase quantity: {max_quantity_based_on_ratio}, Purchase quantity: {purchase_quantity}")

        # 检查槽位可用性
        if self.context.slot_manager.check_availability(time_step, dismiss_time, data_center, total_slots_needed):
            # 创建新的 ServerInfo 并应用变更
            server_id = self.context.id_gen.next_id()
            buy_and_move_info = [ServerMoveInfo(time_step=time_step, target_datacenter=data_center)]
            server_info = ServerInfo(
                server_id=server_id,
                server_generation=server_generation,
                quantity=purchase_quantity,  # 使用动态计算的购买数量
                dismiss_time=dismiss_time,
                buy_and_move_info=buy_and_move_info
            )
            self.context.solution.apply_server_change(server_info)

            # 收集槽位更新
            self.context.slot_manager.push_slot_update(time_step, dismiss_time, data_center, total_slots_needed, 'buy')

            self._print(f"Bought server {server_id} (quantity: {purchase_quantity}, server_generation: {server_generation}, slots_size:{slots_size}) at time {time_step} in datacenter {data_center}, dismiss time: {dismiss_time}")
            return True
        else:
            self._print(f"Not enough slots in datacenter {data_center} for server {server_generation}")
            return False
        
class MoveServerOperation(SA_NeighborhoodOperation):
    pass

class AdjustQuantityOperation(SA_NeighborhoodOperation):
    pass

class AdjustTimeOperation(SA_NeighborhoodOperation):
    pass
    
class RemoveServerOperation(SA_NeighborhoodOperation):
    pass

@dataclass
class SA_status:
    current_score: float = 0.0
    current_temp: float = 0.0
    min_temp:float = 0.0
    alpha: float = 0.0
    max_iter: int = 0
    best_score: float = 0.0
    verbose: bool = False
    seed: int = 0

class SimulatedAnnealing:
    def __init__(self, slot_manager, servers_df, id_gen, solution: DiffSolution, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        self.status = SA_status()
        self.status.current_temp = initial_temp
        self.status.min_temp = min_temp
        self.status.alpha = alpha
        self.status.max_iter = max_iter
        self.status.verbose = verbose
        self.status.seed = seed
        self.id_gen = id_gen

        self.solution = solution
        self.slot_manager: SlotAvailabilityManager = slot_manager

        # 初始化操作上下文
        self.context = OperationContext(
            slot_manager=self.slot_manager,
            servers_df=servers_df,
            id_gen=self.id_gen,
            solution=self.solution,
            verbose=self.status.verbose,
            sa_status=self.status
        )

        # 初始化操作
        self.operations : list[NeighborhoodOperation] = []
        self.operation_probabilities = []
        self.total_weight = 0.0

        # 注册操作
        self.register_operation(
            BuyServerOperation(context=self.context),
            weight=0.4
        )
        self.register_operation(
            MoveServerOperation(context=self.context),
            weight=0.4
        )
        self.register_operation(
            AdjustQuantityOperation(context=self.context),
            weight=0.2
        )
        self.register_operation(
            AdjustTimeOperation(context=self.context),
            weight=0.8
        )
        self.register_operation(
            RemoveServerOperation(context=self.context),
            weight=0.2
        )

        self.best_solution_server_map = copy.deepcopy(self.solution.server_map)
        self.best_score = float('-inf')

    def _print(self, *args, color=None, **kwargs):
        if self.status.verbose:
            # 设置颜色
            if color:
                print(f"{color}{' '.join(map(str, args))}{Style.RESET_ALL}", **kwargs)
            else:
                print(*args, **kwargs)

    def register_operation(self, operation, weight=1.0):
        self.operations.append(operation)
        self.operation_probabilities.append(weight)
        self.total_weight += weight

    def choose_operation(self):
        probabilities = [w / self.total_weight for w in self.operation_probabilities]
        return random.choices(self.operations, weights=probabilities, k=1)[0]

    def generate_neighbor(self):
        self.slot_manager.clear_pending_updates()
        operation = self.choose_operation()
        score, success = operation.execute_and_evaluate()
        return score, success, operation

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def accept_solution(self, accept_prob, new_score):
        if accept_prob >= 1.0 or random.random() < accept_prob:
            self._print(f"Accepted new solution with score {new_score:.5e}", color=Fore.BLUE)
            # 接受新解
            self.solution.commit_server_changes()
            # 检查解是否合法
            if DEBUG:
                result = self.slot_manager.can_accommodate_servers(self.solution.server_map)
                if not result:
                    self._print("New solution is invalid", color=Fore.RED)
                    raise ValueError("New solution is invalid")
            self.slot_manager.apply_pending_updates()
            if new_score > self.status.best_score:
                self.best_solution_server_map = update_best_solution(self.best_solution_server_map, self.solution.server_map)
                self.status.best_score = new_score
                self._print(f"New best solution with score {self.best_score:.5e}", color=Fore.GREEN)
            return True
        else:
            # 拒绝新解并回滚更改
            self.solution.discard_server_changes()
            self._print(f"Rejected new solution with score {new_score:.5e}")
            return False

    def run(self):
        """模拟退火的主循环。"""
        self.status.current_score = self.solution.diff_evaluation()  # 初始评价
        iteration = 0  # 用于记录有效迭代次数
        while iteration < self.status.max_iter:
            self._print(f"<------ Iteration {iteration}, Temperature {self.status.current_temp:.2f} ------>")
            
            new_score, success, operation = self.generate_neighbor()  # 生成一个邻域解
            if success:
                accept_prob = self.acceptance_probability(self.status.current_score, new_score)
                print(f"Iteration: {iteration}. New best solution for {self.status.seed} with score {self.status.best_score:.5e}")
                if self.accept_solution(accept_prob, new_score):
                    self.status.current_score = new_score  # 如果接受，更新当前分数
                    
                # 只有当找到有效邻域解时，才增加迭代次数
                iteration += 1
                self.status.current_temp *= self.status.alpha  # 降低温度
                if self.status.current_temp < self.status.min_temp:
                    break
            else:
                self._print("No valid neighbor found", color=Fore.RED)

        return self.best_solution_server_map, self.status.best_score

def get_my_solution(seed, verbose=False):
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')
    servers[['release_start', 'release_end']] = servers['release_time'].apply(
        lambda x: pd.Series(ast.literal_eval(x))
    )
    slot_manager = SlotAvailabilityManager(datacenters, time_steps=TIME_STEPS, verbose=verbose)
    id_gen = ThreadSafeIDGenerator(start=0)
    solution = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(
        slot_manager=slot_manager,
        servers_df=servers,
        id_gen=id_gen,
        solution=solution,
        seed=seed,
        initial_temp=200000,
        min_temp=100,
        alpha=0.9999,
        max_iter=70000,
        verbose=verbose
    )
    best_solution_server_map, best_score = sa.run()
    print(f'Final best score for {seed}: {best_score:.5e}')
    export_solution_to_json(best_solution_server_map, f"./output/{seed}_{best_score:.5e}.json")
    return best_solution_server_map, best_score

if __name__ == '__main__':
    start = time.time()
    seed = 3329
    best_solution_server_map, best_score = get_my_solution(seed, verbose=False)
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")

```