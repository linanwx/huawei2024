from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
from typing import Dict, Tuple
from colorama import Style
import numpy as np
import pandas as pd
import requests
from idgen import ThreadSafeIDGenerator
from real_diff_evaluation import DiffSolution, ServerInfo

class MonitoringClient:
    def __init__(self, server_url):
        """
        初始化监控客户端
        :param server_url: Dash 应用程序的 HTTP 服务器 URL
        """
        self.server_url = server_url

    def send_numerical_data(self, key, value):
        """
        发送数值类型的数据到监控服务器
        :param key: 数据的标识符
        :param value: 数值类型的数据
        """
        data = {
            'key': key,
            'type': 'numerical',
            'value': value
        }
        self._send_data(data)

    def send_status_data(self, key, status):
        """
        发送状态类型的数据到监控服务器
        :param key: 数据的标识符
        :param status: 'success' 或 'failure'
        """
        if status not in ['success', 'failure']:
            raise ValueError('status must be either "success" or "failure"')
        
        data = {
            'key': key,
            'type': 'status',
            'status': status
        }
        self._send_data(data)

    def _send_data(self, data):
        """
        内部函数，向监控服务器发送数据
        :param data: 要发送的数据字典
        """
        # 使用线程执行发送操作
        threading.Thread(target=self._send_data_in_background, args=(data,)).start()

    def _send_data_in_background(self, data):
        """
        后台执行数据发送的函数
        :param data: 要发送的数据字典
        """
        try:
            response = requests.post(f'{self.server_url}/data', json=data)
            if response.status_code != 200:
                print(f"Error sending data: {response.text}")
        except requests.exceptions.ConnectionError:
            print("Failed to connect to the monitoring server")


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
        if slots_needed == 0:
            return
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
        # 创建一个空的插槽占用情况，与 self.datacenter_slots 的结构一致
        expected_datacenter_slots = {dc: np.zeros(self.time_steps, dtype=int) for dc in self.datacenter_slots.keys()}

        for server_id, server in servers.items():
            if server.slots_size is None:
                raise ValueError(f"服务器 {server_id} 的 slots_size 未定义。")
            if server.dismiss_time < 1 or server.dismiss_time > self.time_steps:
                raise ValueError(f"服务器 {server_id} 的 dismiss_time {server.dismiss_time} 不在有效范围内 (1, {self.time_steps})。")
            if not server.buy_and_move_info:
                raise ValueError(f"服务器 {server_id} 没有任何买入或迁移信息。")

            # 确保 buy_and_move_info 按 time_step 排序
            sorted_moves = sorted(server.buy_and_move_info, key=lambda x: x.time_step)

            # 遍历每一个迁移信息，确定每个阶段的时间范围和目标数据中心
            for i, move in enumerate(sorted_moves):
                start_time = move.time_step
                if i + 1 < len(sorted_moves):
                    end_time = sorted_moves[i + 1].time_step  # 前闭后开: 下一个迁移的时间点作为结束
                else:
                    end_time = server.dismiss_time  # 前闭后开: dismiss_time 作为结束

                if start_time >= end_time:
                    continue  # 无需处理

                if end_time > self.time_steps:
                    end_time = self.time_steps  # 确保不超出时间范围

                dc = move.target_datacenter
                slots_needed = server.quantity * server.slots_size

                # 检查目标数据中心是否存在
                if dc not in expected_datacenter_slots:
                    raise ValueError(f"数据中心 {dc} 不存在。")

                # 在期望的插槽占用中增加占用量
                expected_datacenter_slots[dc][start_time:end_time] += slots_needed

        # 比较期望的插槽占用情况与实际的插槽使用情况
        for dc in self.datacenter_slots.keys():
            # 计算实际占用的插槽数量
            actual_used_slots = self.total_slots[dc] - self.datacenter_slots[dc]
            if not np.all(actual_used_slots >= 0):
                self._print(f"数据中心 {dc} 的插槽占用小于 0。")
                return False
            # 如果期望的占用与实际占用不一致，则返回 False
            if not np.array_equal(expected_datacenter_slots[dc], actual_used_slots):
                self._print(f"数据中心 {dc} 的期望插槽占用与实际不一致。")
                self._print(f"期望的占用：{expected_datacenter_slots[dc]}")
                self._print(f"实际的占用：{actual_used_slots}")
                self._print(f'占用差值：{actual_used_slots - expected_datacenter_slots[dc]}')
                return False

        # 所有数据中心的插槽占用情况都一致
        self._print("所有服务器的插槽占用与当前的插槽使用情况一致。")
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
    
    def calculate_slot_utilization(self):
        total_slots = 0
        used_slots = 0
        utilization_per_datacenter = {}

        for dc, slots in self.datacenter_slots.items():
            # 计算每个数据中心的插槽总数和已使用插槽数
            dc_total_slots = self.total_slots[dc] * self.time_steps
            dc_used_slots = np.sum(self.total_slots[dc] - slots)

            # 计算每个数据中心的利用率
            dc_utilization = dc_used_slots / dc_total_slots if dc_total_slots > 0 else 0
            utilization_per_datacenter[dc] = dc_utilization

            # 统计所有数据中心的总插槽和已使用插槽
            total_slots += dc_total_slots
            used_slots += dc_used_slots

        # 计算整体的插槽利用率
        total_utilization = used_slots / total_slots if total_slots > 0 else 0

        # 返回整体利用率和各个数据中心的利用率
        return total_utilization, utilization_per_datacenter

@dataclass
class SA_status:
    current_score: float = 0.0
    current_temp: float = 0.0
    min_temp:float = 0.0
    alpha: float = 0.0
    max_iter: int = 0
    max_iter_by_temp: int = 0
    current_iter: int = 0
    best_score: float = 0.0
    verbose: bool = False
    seed: int = 0
    monitor: MonitoringClient = None
    best_price_matrix: pd.DataFrame = None

@dataclass
class OperationContext:
    slot_manager: SlotAvailabilityManager
    servers_df: pd.DataFrame
    id_gen: ThreadSafeIDGenerator
    solution: DiffSolution
    sa_status: 'SA_status' 
    verbose: bool = False

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
    def execute_and_evaluate(self) -> Tuple[bool, float]:
        pass