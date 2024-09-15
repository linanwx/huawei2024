import copy
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

        # 处理购买和移动信息
        for move in self.buy_and_move_info:
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

        # dismiss 时间不得超过最大寿命
        if self.buy_and_move_info:
            first_move_time = self.buy_and_move_info[0].time_step
            max_dismiss_time = first_move_time + self.life_expectancy
            if max_dismiss_time < self.dismiss_time:
                self.dismiss_time = max_dismiss_time

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
        self.__capacity_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的需求矩阵
        self.__demand_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的满足矩阵
        self.__satisfaction_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化每一个时间步骤的服务器寿命
        self.__lifespan = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的寿命百分比总和
        self.__lifespan_percentage_sum = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的服务器数量
        self.__fleetsize = np.zeros(TIME_STEPS, dtype=float)
        # 初始化平均寿命百分比
        self.__average_lifespan = np.zeros(TIME_STEPS, dtype=float)
        # 初始化每一个时间步骤的成本
        self.__cost = np.zeros(TIME_STEPS, dtype=float)
        # 差分黑板，用于暂存变动
        self.__blackboard: DiffBlackboard = None
        # 当前的差分信息
        self.__diff_info: ServerInfo = None
        # 初始化利用率矩阵
        self.__utilization_matrix = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)), dtype=float)
        # 初始化平均利用率
        self.__average_utilization = np.zeros(TIME_STEPS, dtype=float)

        self._load_demand_data('./data/demand.csv', seed)
        self.__verbose = verbose

        self.__cost_details = {t: {'purchase_cost': 0, 'energy_cost': 0, 'maintenance_cost': 0} for t in range(TIME_STEPS)}
        self.__capacity_combinations = {t: {} for t in range(TIME_STEPS)}

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
        self.__price_matrix = np.zeros((num_latencies, num_servers), dtype=float)
        for latency_key, latency_idx in LATENCY_SENSITIVITY_MAP.items():
            for server_key, server_idx in SERVER_GENERATION_MAP.items():
                price_key = (server_key, latency_key)
                selling_price = self.__selling_price_dict.get(price_key, 0)
                self.__price_matrix[latency_idx, server_idx] = selling_price

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
            self.__demand_matrix[time_indices, latency_idx, server_indices] = demand_values

    def _load_data(self):
        """
        加载售价数据
        """
        # 加载售价数据
        self.__selling_prices_df = pd.read_csv('./data/selling_prices.csv')
        self.__selling_prices_df['server_generation'] = self.__selling_prices_df['server_generation'].astype(str)
        self.__selling_prices_df['latency_sensitivity'] = self.__selling_prices_df['latency_sensitivity'].astype(str)
        # 建立 (server_generation, latency_sensitivity) 的售价字典
        self.__selling_price_dict = self.__selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()

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
        self.__lifespan = self.__blackboard.lifespan
        self.__lifespan_percentage_sum = self.__blackboard.lifespan_percentage_sum
        self.__fleetsize = self.__blackboard.fleetsize
        self.__capacity_matrix = self.__blackboard.capacity_matrix
        self.__cost = self.__blackboard.cost
        self.__utilization_matrix = self.__blackboard.utilization_matrix
        self.__average_utilization = self.__blackboard.average_utilization
        self.__average_lifespan = self.__blackboard.average_lifespan
        self.__satisfaction_matrix = self.__blackboard.satisfaction_matrix
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
        if self.__blackboard is None:
            # Initialize blackboard
            self.__blackboard = DiffBlackboard(
                lifespan=self.__lifespan.copy(),
                lifespan_percentage_sum=self.__lifespan_percentage_sum.copy(),
                fleetsize=self.__fleetsize.copy(),
                capacity_matrix=self.__capacity_matrix.copy(),
                cost=self.__cost.copy(),
                satisfaction_matrix=self.__satisfaction_matrix.copy(),
                changed_capacity_indices=set(),
                utilization_matrix=self.__utilization_matrix.copy(),
                average_utilization=self.__average_utilization.copy(),
                average_lifespan=self.__average_lifespan.copy(),
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
        time_start = diff_info.buy_and_move_info[0].time_step
        time_end = diff_info.dismiss_time
        lifespan_data = blackboard.lifespan
        lifespan_percentage_sum = blackboard.lifespan_percentage_sum

        # 确保时间范围在数组索引范围内
        time_start = max(0, time_start)
        time_end = min(TIME_STEPS, time_end)

        if time_end > time_start:
            # 计算寿命增量
            lifespan_steps = np.arange(1, time_end - time_start + 1, dtype=float)
            increments = lifespan_steps * diff_info.quantity * sign
            # 更新寿命数据
            lifespan_data[time_start:time_end] += increments
            # 计算寿命百分比增量
            life_expectancy = diff_info.life_expectancy
            lifespan_percentages = (lifespan_steps / life_expectancy) * diff_info.quantity * sign
            # 更新寿命百分比总和
            lifespan_percentage_sum[time_start:time_end] += lifespan_percentages
            # 记录受影响的时间步骤
            blackboard.changed_time_steps.update(range(time_start, time_end))
        else:
            raise ValueError("结束时间必须大于开始时间。")

    def _update_fleet_size(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新服务器数量。
        """
        time_start = diff_info.buy_and_move_info[0].time_step
        time_end = diff_info.dismiss_time

        fleet_size = blackboard.fleetsize

        # 确保时间范围在数组索引范围内
        time_start = max(0, time_start)
        time_end = min(TIME_STEPS, time_end)

        # 更新服务器数量
        fleet_size[time_start:time_end] += diff_info.quantity * sign

        # 记录受影响的时间步骤
        blackboard.changed_time_steps.update(range(time_start, time_end))

    def _update_buy_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新购买成本。
        """
        cost_array = blackboard.cost
        purchase_time = diff_info.buy_and_move_info[0].time_step
        purchase_cost = diff_info.purchase_price * diff_info.quantity * sign
        if 0 <= purchase_time < TIME_STEPS:
            cost_array[purchase_time] += purchase_cost
            if self.__verbose:
                self.__cost_details[purchase_time]['purchase_cost'] += purchase_cost

    def _update_moving_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新迁移成本。
        """
        cost_array = blackboard.cost
        for move_info in diff_info.buy_and_move_info[1:]:  # 跳过第一个购买信息
            move_time = move_info.time_step
            moving_cost = diff_info.cost_of_moving * diff_info.quantity * sign
            if 0 <= move_time < TIME_STEPS:
                cost_array[move_time] += moving_cost
            print(f"迁移费用 (时间步 {move_time}): {moving_cost}")

    def _update_energy_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新能耗成本。
        """
        cost_array = blackboard.cost
        move_info_list = diff_info.buy_and_move_info
        for i in range(len(move_info_list)):
            current_move = move_info_list[i]
            time_start_energy = current_move.time_step
            if i + 1 < len(move_info_list):
                next_move = move_info_list[i + 1]
                time_end_energy = next_move.time_step
            else:
                time_end_energy = diff_info.dismiss_time

            # 确保时间索引在有效范围内
            time_start_energy = max(0, time_start_energy)
            time_end_energy = min(TIME_STEPS, time_end_energy)

            # 计算能耗成本
            energy_consumption = diff_info.energy_consumption * diff_info.quantity  # 总能耗
            cost_of_energy = current_move.cost_of_energy  # 数据中心的电力成本
            energy_cost_per_time_step = energy_consumption * cost_of_energy * sign  # 每个时间步的能耗成本

            # 更新成本数组
            if time_start_energy < time_end_energy:
                cost_array[time_start_energy:time_end_energy] += energy_cost_per_time_step
                if self.__verbose:
                    for t in range(time_start_energy, time_end_energy):
                                    self.__cost_details[t]['energy_cost'] += energy_cost_per_time_step

    def _update_maintenance_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新维护成本。
        """
        cost_array = blackboard.cost
        time_start = diff_info.buy_and_move_info[0].time_step
        time_end = diff_info.dismiss_time

        # 确保时间索引在有效范围内
        time_start = max(0, time_start)
        time_end = min(TIME_STEPS, time_end)

        # 提取服务器的属性
        xhat = diff_info.life_expectancy
        b = diff_info.average_maintenance_fee
        quantity = diff_info.quantity

        # 计算服务器在生命周期内每个时间步的寿命
        lifespan_steps = np.arange(1, time_end - time_start + 1)

        # 计算 ratio
        ratio = (1.5 * lifespan_steps) / xhat
        # 避免 log2(0) 的情况
        ratio = np.maximum(ratio, 1e-6)

        # 计算维护成本
        maintenance_cost_per_time_step = b * (1 + ratio * np.log2(ratio)) * quantity * sign

        # 更新成本数组
        cost_array[time_start:time_end] += maintenance_cost_per_time_step

        if self.__verbose:
            for t in range(time_start, time_end):
                self.__cost_details[t]['maintenance_cost'] += maintenance_cost_per_time_step[t - time_start]

    def _recalculate_satisfaction(self, blackboard: DiffBlackboard):
        """
        根据容量变化的区域，重新计算需求满足数组，并更新利用率矩阵。
        """
        # 获取所有受影响的索引
        changed_indices = list(blackboard.changed_capacity_indices)
        if not changed_indices:
            return  # 如果没有变化，直接返回

        # 将索引列表转换为数组，便于矢量化操作
        changed_indices = np.array(changed_indices)
        time_steps = changed_indices[:, 0]
        latency_idxs = changed_indices[:, 1]
        server_generation_idxs = changed_indices[:, 2]

        # 获取需求和容量数据
        demand_values = self.__demand_matrix[time_steps, latency_idxs, server_generation_idxs]
        capacity_values = blackboard.capacity_matrix[time_steps, latency_idxs, server_generation_idxs]

        # 计算需求满足值
        satisfaction_values = np.minimum(demand_values, capacity_values)

        # 更新 blackboard 的 satisfaction_matrix
        blackboard.satisfaction_matrix[time_steps, latency_idxs, server_generation_idxs] = satisfaction_values

        # 计算利用率
        # 注意要避免除以零的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization_values = np.where(
                capacity_values > 0,
                satisfaction_values / capacity_values,
                0.0
            )
        # 更新 blackboard 的 utilization_matrix
        blackboard.utilization_matrix[time_steps, latency_idxs, server_generation_idxs] = utilization_values

    def _adjust_capacity_by_failure_rate_approx(self, x, avg_failure_rate=FAILURE_RATE):
        return x * (1 - avg_failure_rate)

    def _update_capacity(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新容量矩阵，并记录容量变化的索引。
        """
        capacity_matrix = blackboard.capacity_matrix
        capacity = diff_info.capacity * diff_info.quantity
        capacity = self._adjust_capacity_by_failure_rate_approx(capacity) * sign

        move_info_list = diff_info.buy_and_move_info
        server_generation_idx = SERVER_GENERATION_MAP[diff_info.server_generation]
        for i in range(len(move_info_list)):
            current_move = move_info_list[i]
            time_start = current_move.time_step
            if i + 1 < len(move_info_list):
                next_move = move_info_list[i + 1]
                time_end = next_move.time_step
            else:
                time_end = diff_info.dismiss_time
            # 确保时间索引在有效范围内
            time_start = max(0, time_start)
            time_end = min(TIME_STEPS, time_end)
            latency_sensitivity = current_move.latency_sensitivity

            # 更新容量矩阵
            capacity_matrix[time_start:time_end, latency_sensitivity, server_generation_idx] += capacity

            # 记录容量变化的索引
            for t in range(time_start, time_end):
                blackboard.changed_capacity_indices.add((t, latency_sensitivity, server_generation_idx))

    def _calculate_average_utilization(self, blackboard: DiffBlackboard):
        utilization_matrix = blackboard.utilization_matrix
        changed_indices = list(blackboard.changed_capacity_indices)

        if not changed_indices:
            # 如果没有变化，直接返回之前计算的平均利用率
            return blackboard.average_utilization

        # 将索引列表转换为数组
        changed_indices = np.array(changed_indices)
        time_steps = changed_indices[:, 0]

        # 获取所有受影响的时间步骤的集合
        affected_time_steps = np.unique(time_steps)

        # 对于受影响的时间步骤，重新计算 valid_counts 和 utilization_sums
        for t in affected_time_steps:
            valid_capacity_mask_t = blackboard.capacity_matrix[t] > 0
            valid_counts_t = np.sum(valid_capacity_mask_t)
            utilization_sums_t = np.sum(utilization_matrix[t])

            if valid_counts_t > 0:
                blackboard.average_utilization[t] = utilization_sums_t / valid_counts_t
            else:
                blackboard.average_utilization[t] = 0.0

        return blackboard.average_utilization

    def _calculate_average_lifespan(self, blackboard: DiffBlackboard):
        lifespan_percentage_sum = blackboard.lifespan_percentage_sum
        fleetsize = blackboard.fleetsize
        changed_time_steps = np.array(list(blackboard.changed_time_steps))

        if len(changed_time_steps) == 0:
            return blackboard.average_lifespan

        # 计算平均寿命百分比，避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            average_lifespan = np.divide(
                lifespan_percentage_sum,
                fleetsize,
                out=np.zeros_like(lifespan_percentage_sum),
                where=fleetsize != 0
            )

        # 只更新受影响的时间步
        blackboard.average_lifespan[changed_time_steps] = average_lifespan[changed_time_steps]

        return blackboard.average_lifespan

    def _calculate_revenue(self, blackboard: DiffBlackboard):
        revenue = np.sum(blackboard.satisfaction_matrix * self.__price_matrix, axis=(1, 2))
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
                    '购买费用': round(self.__cost_details[t]['purchase_cost'], 2),
                    '能耗费用': round(self.__cost_details[t]['energy_cost'], 2),
                    '维护费用': round(self.__cost_details[t]['maintenance_cost'], 2),
                    '总费用': round(blackboard.cost[t], 2),
                    '总收入': round(self._calculate_revenue(blackboard)[t], 2),
                })
                self.__print(f"Time Step {t + 1} - Capacity Combinations: {self.__capacity_combinations[t]}")
        return evaluation_result
    
    def diff_evaluation(self):
        """
        差分评估函数
        """
        if self.__blackboard is None:
            raise Exception("No blackboard data. Apply server changes before evaluation.")
        # Recalculate satisfaction
        self._recalculate_satisfaction(self.__blackboard)

        evaluation_result = self._final_calculation(self.__blackboard)
        return evaluation_result
