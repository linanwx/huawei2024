from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from idgen import ThreadSafeIDGenerator
from real_diff_evaluation import DiffSolution, ServerInfo

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

@dataclass
class OperationContext:
    slot_manager: SlotAvailabilityManager
    servers_df: pd.DataFrame
    id_gen: ThreadSafeIDGenerator
    solution: DiffSolution
    sa_status: 'SA_status' 
    verbose: bool = False

