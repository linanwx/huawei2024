import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List
from evaluation import get_actual_demand

TIME_STEPS = 168  # 时间下标从0到167
FAILURE_RATE = 0.0725  # 故障率

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
    datacenter_id: str                   # 购买时机房ID
    move_info: List[ServerMoveInfo]      # 服务器购买和迁移信息列表
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
        # 初始化每一个时间步骤的服务器数量
        self.__fleetsize = np.zeros(TIME_STEPS, dtype=float)
        # 初始化平均寿命
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
        self.verbose = verbose

    def __print(self, message):
        """
        根据verbose的值决定是否打印消息。
        :param message: 要打印的消息。
        """
        if self.verbose:
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
        加载服务器、数据中心和售价数据，并构建快速查找的数据结构。
        """
        # 加载服务器数据
        self.__servers_df = pd.read_csv('./data/servers.csv')
        self.__servers_df['server_generation'] = self.__servers_df['server_generation'].astype(str)
        self.__server_info_dict = self.__servers_df.set_index('server_generation').to_dict('index')

        # 加载数据中心数据
        self.__datacenters_df = pd.read_csv('./data/datacenters.csv')
        self.__datacenters_df['datacenter_id'] = self.__datacenters_df['datacenter_id'].astype(str)
        self.__datacenter_info_dict = self.__datacenters_df.set_index('datacenter_id').to_dict('index')

        # 加载售价数据
        self.__selling_prices_df = pd.read_csv('./data/selling_prices.csv')
        self.__selling_prices_df['server_generation'] = self.__selling_prices_df['server_generation'].astype(str)
        self.__selling_prices_df['latency_sensitivity'] = self.__selling_prices_df['latency_sensitivity'].astype(str)
        # 建立 (server_generation, latency_sensitivity) 的售价字典
        self.__selling_price_dict = self.__selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()

    def set_server_diff(self, diff_info: ServerInfo):
        """
        对解进行服务器变动的操作，填充必要的信息。
        """
        server_gen = diff_info.server_generation
        server_info = self.__server_info_dict.get(server_gen)
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
            datacenter_info = self.__datacenter_info_dict.get(datacenter_id)
            if datacenter_info is None:
                raise ValueError(f"数据中心 {datacenter_id} 未找到。")

            move.cost_of_energy = datacenter_info['cost_of_energy']
            latency_sensitivity_str = datacenter_info['latency_sensitivity']
            move.latency_sensitivity = LATENCY_SENSITIVITY_MAP[latency_sensitivity_str]

            # 查找售价
            key = (server_gen, latency_sensitivity_str)
            selling_price = self.__selling_price_dict.get(key)
            if selling_price is None:
                raise ValueError(f"售价未找到，服务器代次：{server_gen}，时延敏感性：{latency_sensitivity_str}")
            move.selling_price = selling_price
        # dismiss 时间不得超过最大寿命
        if diff_info.move_info[0].time_step + diff_info.life_expectancy < diff_info.dismiss_time:
            diff_info.dismiss_time = diff_info.move_info[0].time_step + diff_info.life_expectancy

        self.__diff_info = diff_info

    def commit_server_changes(self):
        """
        将当前对服务器的变动操作，应用到解中。
        """
        # 将 blackboard 中的数据正式写入到解的内部状态
        self.__lifespan = self.__blackboard.lifespan
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

    def _apply_change(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        应用指定的差分信息，更新黑板数据，不修改解的内部状态。
        """
        # 调整时间步骤寿命
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
        更新服务器寿命。
        """
        time_start = diff_info.move_info[0].time_step
        time_end = diff_info.dismiss_time
        lifespan_data = blackboard.lifespan

        # 确保时间范围在数组索引范围内
        time_start = max(0, time_start)
        time_end = min(TIME_STEPS, time_end)

        if time_end > time_start:
            # 计算寿命增量
            lifespan_steps = np.arange(1, time_end - time_start + 1, dtype=float)
            increments = lifespan_steps * diff_info.quantity * sign

            # 更新寿命数据
            lifespan_data[time_start:time_end] += increments

            # 记录受影响的时间步骤
            blackboard.changed_time_steps.update(range(time_start, time_end))
        else:
            raise ValueError("结束时间必须大于开始时间。")


    def _update_fleet_size(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新服务器数量。
        """
        time_start = diff_info.move_info[0].time_step
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
        purchase_time = diff_info.move_info[0].time_step
        purchase_cost = diff_info.purchase_price * diff_info.quantity * sign
        if 0 <= purchase_time < TIME_STEPS:
            cost_array[purchase_time] += purchase_cost

    def _update_moving_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新迁移成本。
        """
        cost_array = blackboard.cost
        for move_info in diff_info.move_info[1:]:  # 跳过第一个购买信息
            move_time = move_info.time_step
            moving_cost = diff_info.cost_of_moving * diff_info.quantity * sign
            if 0 <= move_time < TIME_STEPS:
                cost_array[move_time] += moving_cost

    def _update_energy_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新能耗成本。
        """
        cost_array = blackboard.cost
        move_info_list = diff_info.move_info
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

    def _update_maintenance_cost(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign: int):
        """
        更新维护成本。
        """
        cost_array = blackboard.cost
        time_start = diff_info.move_info[0].time_step
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

        # 可以添加日志来检查利用率
        print(f"Time steps affected: {time_steps}")
        print(f"  Utilization values: {utilization_values}")


    def _adjust_capacity_by_failure_rate_approx(self, x, avg_failure_rate=FAILURE_RATE):
        """
        近似调整容量以考虑故障率。
        """
        return x * (1 - avg_failure_rate)
    
    def _check_negative_capacity(self, capacity_matrix, time_start, time_end, latency_sensitivity, server_generation_idx):
        """
        检查指定索引范围的容量是否为负，如果是，则抛出错误。
        """
        sub_matrix = capacity_matrix[time_start:time_end, latency_sensitivity, server_generation_idx]
        if (sub_matrix < 0).any():
            raise ValueError("容量矩阵中存在负值，更新操作无效。")

    def _update_capacity(self, blackboard: DiffBlackboard, diff_info: ServerInfo, sign=1):
        """
        更新容量矩阵，并记录容量变化的索引。
        """
        capacity_matrix = blackboard.capacity_matrix
        capacity = diff_info.capacity * diff_info.quantity
        capacity = self._adjust_capacity_by_failure_rate_approx(capacity) * sign

        move_info_list = diff_info.move_info
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
        
        self._check_negative_capacity(capacity_matrix, time_start, time_end, latency_sensitivity, server_generation_idx)

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

            # 日志输出，检查有效容量掩码和有效服务器数量
            print(f"Time step {t}:")
            print(f"  Valid capacity mask: {valid_capacity_mask_t}")
            print(f"  Valid counts (valid server count): {valid_counts_t}")
            print(f"  Utilization sums: {utilization_sums_t}")

            if valid_counts_t > 0:
                blackboard.average_utilization[t] = utilization_sums_t / valid_counts_t
            else:
                blackboard.average_utilization[t] = 0.0

            # 输出计算结果
            print(f"  Average utilization for time step {t}: {blackboard.average_utilization[t]}\n")

        return blackboard.average_utilization


    
    def _calculate_average_lifespan(self, blackboard: DiffBlackboard):
        lifespan = blackboard.lifespan
        fleetsize = blackboard.fleetsize
        changed_time_steps = np.array(list(blackboard.changed_time_steps))

        if len(changed_time_steps) == 0:
            return blackboard.average_lifespan

        # 初始化数组
        total_lifespan_percentages = np.zeros(TIME_STEPS)
        total_quantities = np.zeros(TIME_STEPS)

        # 遍历所有服务器，计算它们在每个时间步的寿命比例和数量
        for server_info in self.server_map.values():
            life_expectancy = server_info.life_expectancy
            quantity = server_info.quantity

            # 获取服务器的启动时间和下线时间
            time_start = server_info.move_info[0].time_step
            time_end = server_info.dismiss_time

            # 创建服务器的活动时间范围
            server_time_steps = np.arange(time_start, time_end)
            # 计算当前寿命（第一个时间步寿命为1）
            current_lifespans = server_time_steps - time_start + 1
            # 计算寿命比例
            lifespan_percentages = current_lifespans / life_expectancy
            # 将寿命比例乘以服务器数量，累加到总寿命比例数组中
            total_lifespan_percentages[time_start:time_end] += lifespan_percentages * quantity
            # 累加服务器数量
            total_quantities[time_start:time_end] += quantity

        # 计算平均寿命比例，避免除以零
        with np.errstate(divide='ignore', invalid='ignore'):
            average_lifespan = np.divide(
                total_lifespan_percentages,
                total_quantities,
                out=np.zeros_like(total_lifespan_percentages),
                where=total_quantities != 0
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
        
        # 计算每个时间步的平均寿命
        average_lifespan = self._calculate_average_lifespan(blackboard)
        
        # 计算每个时间步的利润
        profit = self._calculate_profit(blackboard)
        
        # 计算每个时间步的乘积： 平均利用率 * 平均寿命 * 利润
        stepwise_product = average_utilization * average_lifespan * profit
        
        # 计算所有时间步乘积的总和
        evaluation_result = np.sum(stepwise_product)

        if self.verbose:
            for t in range(TIME_STEPS):
                print({
                    'time-step': t + 1,
                    'U': round(average_utilization[t], 2),
                    'L': round(average_lifespan[t], 2),
                    'P': round(profit[t], 2),
                    'Size': int(blackboard.fleetsize[t]),
                })
        
        return evaluation_result

    def diff_evaluation(self):
        """
        找到当前解中需要变动的服务器ID
        如果当前解中存在该服务器
            移除对应时间步骤的总寿命
            移除对应时间步骤的服务器总数量
            移除对应时间步骤的对应时延-代数服务器容量
            在成本中，移除对应步骤的购买成本 移动成本 维护成本

        针对重新设置的该服务器参数
            增加对应时间步骤的总寿命
            增加对应时间步骤的服务器总数量
            增加对应时间步骤的时延-代数服务器容量
            在成本中，增加对应步骤的购买成本 移动成本 维护成本

        更新所有时间步骤 受到影响的（特定时间步骤，特定服务器代数，特定时延敏感）每步骤 满足量
        更新所有步骤 受到影响的 服务器组合数量
        
        更新所有步骤的 对应 时延-代数的利润

        计算每步平均寿命
        计算每步骤利用率
        计算每步骤总利润
        """
        new_diff_info = self.__diff_info
        original_server_info = self.server_map.get(new_diff_info.server_id)

        # 初始化黑板
        blackboard = DiffBlackboard(
            lifespan=self.__lifespan.copy(),
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

        # 逆转旧的服务器影响（如果存在）
        if original_server_info is not None:
            self._apply_change(blackboard, original_server_info, sign=-1)

        # 应用新的服务器变动
        if new_diff_info.quantity > 0:
            self._apply_change(blackboard, new_diff_info, sign=1)

        # 重新计算需求满足数组
        self._recalculate_satisfaction(blackboard)

        # 执行最终计算，得到评估结果
        evaluation_result = self._final_calculation(blackboard)

        # 保存黑板
        self.__blackboard = blackboard
        return evaluation_result
