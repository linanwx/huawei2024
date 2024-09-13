import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

TIME_STEPS = 168  # 时间下标从1到168 下标从0到167
NUM_latency_sensitivities = 3
NUM_server_generations = 7

# 时延敏感性
LATENCY_SENSITIVITY_MAP = {'low': 0, 'medium': 1, 'high': 2}

# 服务器代次
SERVER_GENERATION_MAP = {
    'CPU.S1': 0,
    'CPU.S2': 1,
    'CPU.S3': 2,
    'CPU.S4': 3,
    'GPU.S1': 4,
    'GPU.S2': 5,
    'GPU.S3': 6
}

@dataclass
class ServerMoveInfo:
    """
    表示服务器移动信息
    
    Attributes:
        time_step (int): 迁移操作发生的时间点
        target_datacenter (str): 目标机房ID
    """
    # 需要传入的属性
    time_step: int
    target_datacenter: str
    # 其他联查属性 不需要传入
    cost_of_energy: float               # 能耗成本
    latency_sensitivity: int            # 时延敏感性
    selling_price: float                # 利润价格

@dataclass
class ServerInfo:
    """
    服务器信息
    """
    # 需要传入的属性
    server_id: str                      # 服务器ID
    dismiss_time: int                   # 提前下线时间 最小为 2 最大为 TIME_STEPS + 1 正常下线时间是购买时间 + life_expectancy
    datacenter_id: str                  # 购买时机房ID
    move_info: list[ServerMoveInfo]     # 服务器购买和迁移信息列表 第一个元素代表购买时间和购买的机房，后续代表迁移时间和迁移的机房
    quantity: int                       # 服务器数量
    server_generation: str              # 服务器代次
    # 其他联查属性 不需要传入
    purchase_price: float = None        # 购买价格
    slots_size: int = None              # 服务器槽位大小
    energy_consumption: int = None      # 能耗
    capacity: int = None                # 容量
    life_expectancy: int = None         # 最大寿命
    cost_of_moving: float = None        # 迁移成本
    average_maintenance_fee: float = None  # 平均维护费用

@dataclass
class DiffBlackboard:
    lifespan: np.ndarray
    fleetsize: np.ndarray

class DiffSolution:
    def __init__(self):
        # 加载数据
        self.load_data()
        # 初始化一个空解
        self.server_map: Dict[str, ServerInfo] = {}
        # 初始化每一个时间步骤 服务器容量
        self.capacity_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=int)
        # 初始化每一个时间步骤 服务器寿命
        self.lifespan = np.zeros(TIME_STEPS, dtype=int)
        # 初始化每一个时间步骤 服务器数量
        self.fleetsize = np.zeros(TIME_STEPS, dtype=int)
        # 初始化每一个时间步骤 成本
        self.cost = np.zeros(TIME_STEPS, dtype=float)
    def load_data(self):
        """
        加载服务器、数据中心和售价数据，并构建快速查找的数据结构。
        """
        # 加载服务器数据
        self.servers_df = pd.read_csv('./data/servers.csv')
        self.servers_df['server_generation'] = self.servers_df['server_generation'].astype(str)
        self.server_info_dict = self.servers_df.set_index('server_generation').to_dict('index')

        # 加载数据中心数据
        self.datacenters_df = pd.read_csv('./data/datacenters.csv')
        self.datacenters_df['datacenter_id'] = self.datacenters_df['datacenter_id'].astype(str)
        self.datacenter_info_dict = self.datacenters_df.set_index('datacenter_id').to_dict('index')

        # 加载售价数据
        self.selling_prices_df = pd.read_csv('./data/selling_prices.csv')
        self.selling_prices_df['server_generation'] = self.selling_prices_df['server_generation'].astype(str)
        self.selling_prices_df['latency_sensitivity'] = self.selling_prices_df['latency_sensitivity'].astype(str)
        # 建立 (server_generation, latency_sensitivity) 的售价字典
        self.selling_price_dict = self.selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()
        
    def set_server_diff(self, diff_info: ServerInfo):
        """
        对解进行服务器变动的操作
        
        Parameters:
            diff_info (ServerChangeInfo): 服务器的变动信息
        """
        server_gen = diff_info.server_generation
        server_info = self.server_info_dict.get(server_gen)
        if server_info is None:
            raise ValueError(f"服务器代次 {server_gen} 未找到。")
        # 填充服务器信息
        diff_info.purchase_price = server_info['purchase_price']
        diff_info.slots_size = server_info['slots_size']
        diff_info.energy_consumption = server_info['energy_consumption']
        diff_info.capacity = server_info['capacity']
        diff_info.life_expectancy = server_info['life_expectancy']
        diff_info.cost_of_moving = server_info['cost_of_moving']
        diff_info.average_maintenance_fee = server_info['average_maintenance_fee']
        # 处理购买和移动信息
        for move in diff_info.move_info:
            datacenter_id = move.target_datacenter
            datacenter_info = self.datacenter_info_dict.get(datacenter_id)
            if datacenter_info is None:
                raise ValueError(f"数据中心 {datacenter_id} 未找到。")

            move.cost_of_energy = datacenter_info['cost_of_energy']
            latency_sensitivity_str = datacenter_info['latency_sensitivity']
            move.latency_sensitivity = LATENCY_SENSITIVITY_MAP[latency_sensitivity_str]

            # 查找售价
            key = (server_gen, latency_sensitivity_str)
            selling_price = self.selling_price_dict.get(key)
            if selling_price is None:
                raise ValueError(f"售价未找到，服务器代次：{server_gen}, 时延敏感性：{latency_sensitivity_str}")
            move.selling_price = selling_price

        # 将更新后的服务器信息存储到 server_map 中
        self.server_map[diff_info.server_id] = diff_info

    def commit_server_changes(self):
        """
        将当前对服务器的变动操作，应用到解中
        """
        pass

    def _update_lifespan(self, lifespan_data: np.ndarray, time_start, time_end, quantity: int, initial_lifespan=1, sign=1) -> np.ndarray:
        time_start = max(0, time_start)
        time_end = min(len(lifespan_data), time_end)
        # 生成从 initial_lifespan 开始的增量序列，并乘以数量
        increments = np.arange(initial_lifespan, initial_lifespan + (time_end - time_start)) * quantity
        # 更新寿命数据
        updated_lifespan_data = np.copy(lifespan_data)
        updated_lifespan_data[time_start:time_end] += increments * sign
        return updated_lifespan_data

    def _update_fleet_size(self, fleet_size: np.ndarray, buy_time: int, dismiss_time: int, quantity: int, sign=1) -> np.ndarray:
        # 确保时间范围在数组索引范围内
        time_start = max(0, buy_time)
        time_end = min(TIME_STEPS, dismiss_time)
        
        # 更新指定时间范围内的服务器数量
        updated_fleet_size = np.copy(fleet_size)
        updated_fleet_size[time_start:time_end] += quantity * sign
        return updated_fleet_size

    def _apply_change(self, diff_info: ServerInfo, blackboard: DiffBlackboard, sign=1):
        # 调整时间索引
        time_start = diff_info.move_info[0].time_step - 1
        time_end = diff_info.dismiss_time - 1

        # 调整时间步骤寿命
        blackboard.lifespan = self._update_lifespan(
            self.lifespan,
            time_start,
            time_end,
            quantity=diff_info.quantity,
            sign=sign
        )
        # 调整时间步骤服务器数量
        blackboard.fleetsize = self._update_fleet_size(
            self.fleetsize,
            time_start,
            time_end,
            quantity=diff_info.quantity,
            sign=sign
        )
        # 调整容量矩阵
        self._update_capacity(diff_info, sign=1)
        # 购买成本
        purchase_time = time_start
        purchase_cost = diff_info.purchase_price * diff_info.quantity
        if 0 <= purchase_time < TIME_STEPS:
            self.cost[purchase_time] += purchase_cost * sign

        # 移动成本
        for move_info in diff_info.move_info[1:]:  # 跳过第一个购买信息
            move_time = move_info.time_step - 1
            moving_cost = diff_info.cost_of_moving * diff_info.quantity
            if 0 <= move_time < TIME_STEPS:
                self.cost[move_time] += moving_cost * sign


    def adjust_capacity_by_failure_rate_approx(self, x, avg_failure_rate=0.0725):
        return int(x * (1 - avg_failure_rate))

    def _update_capacity(self, diff_info: ServerInfo, sign=1):
        capacity = diff_info.capacity * diff_info.quantity
        capacity = self.adjust_capacity_by_failure_rate_approx(capacity)

        move_info_list = diff_info.move_info
        server_generation_idx = SERVER_GENERATION_MAP[diff_info.server_generation]
        for i in range(len(move_info_list)):
            current_move = move_info_list[i]
            time_start = current_move.time_step - 1  # 调整为数组索引
            if i + 1 < len(move_info_list):
                next_move = move_info_list[i + 1]
                time_end = next_move.time_step - 1  # 调整为数组索引
            else:
                time_end = diff_info.dismiss_time - 1  # 调整为数组索引
            # 确保时间索引在有效范围内
            time_start = max(0, time_start)
            time_end = min(TIME_STEPS, time_end)
            latency_sensitivity = current_move.latency_sensitivity
            self.capacity_matrix[time_start:time_end, latency_sensitivity, server_generation_idx] += capacity * sign

    def _final_calcualtion(self):
        pass

    def diff_evaluation(self):
        """
        对当前解进行差分评估
        
        找到当前解中需要变动的服务器ID
        若存在针对其已经有的 buy操作 和 move操作：
            移除对应时间步骤的总寿命
            移除对应时间步骤的服务器总数量
            移除对应时间步骤的对应时延-代数服务器容量

            在成本中，移除对应步骤的购买成本
            在成本中，移除对应步骤的移动成本
            在成本中，移除对应步骤的维护成本

        针对新增的服务器操作：
            增加对应时间步骤的总寿命
            增加对应时间步骤的服务器总数量
            增加对应时间步骤的时延-代数服务器容量

            在成本中，增加对应步骤的购买成本
            在成本中，增加对应步骤的移动成本
            在成本中，增加对应步骤的维护成本

        更新所有步骤 时延-代数服务器组合数量
        更新所有时间步骤的对应时延-代数每步骤供应量

        更新所有步骤的 对应 时延-代数的利润

        计算每步平均寿命
        计算每步骤利用率
        计算每步骤总利润
        """
        pass


