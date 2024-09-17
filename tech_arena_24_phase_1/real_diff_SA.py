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
from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo
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
        # 初始化一个临时的插槽剩余容量快照，用总容量填充
        temp_datacenter_slots = {dc: np.full(self.time_steps, self.total_slots[dc], dtype=int) for dc in self.datacenter_slots.keys()}

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
                if dc not in self.total_slots:
                    raise ValueError(f"数据中心 {dc} 不存在。")

                # 在临时插槽中减少剩余容量
                temp_datacenter_slots[dc][start_time:end_time] -= slots_needed

                # 检查是否超过容量（即剩余容量是否为负数）
                if np.any(temp_datacenter_slots[dc][start_time:end_time] < 0):
                    self._print(f"服务器 {server_id} 在数据中心 {dc} 的时间范围 {start_time} 到 {end_time} 超过插槽容量。需要 {slots_needed}，当前剩余容量 {temp_datacenter_slots[dc][start_time:end_time]}")                    
                    return False

        # 如果所有服务器都可以被容纳
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

class NeighborhoodOperation(ABC):
    def __init__(self, context: OperationContext):
        self.context = context

    def _print(self, *args, **kwargs):
        if self.context.verbose:
            print(*args, **kwargs)

    @abstractmethod
    def execute(self):
        pass

class BuyServerOperation(NeighborhoodOperation):
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
            self._print(f"Not enough slots in datacenter {data_center} for server {server_generation}")
            return False

        # 计算最大购买数量，基于 MAX_PURCHASE_RATIO
        max_quantity_based_on_ratio = int((max_available_slots // slots_size) * self.MAX_PURCHASE_RATIO)
        max_quantity = max(1, max_quantity_based_on_ratio)  # 确保至少购买1个

        if max_quantity_based_on_ratio >= 1:
            purchase_quantity = random.randint(1, min(int(max_quantity_based_on_ratio), max_quantity))
        else:
            purchase_quantity = 1

        # 计算所需插槽
        total_slots_needed = purchase_quantity * slots_size
        self._print(f"Max available slots: {max_available_slots}, Max purchase quantity: {max_quantity}, Purchase quantity: {purchase_quantity}")

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
        
class MoveServerOperation(NeighborhoodOperation):
    def execute(self):
        if not self.context.solution.server_map:
            self._print("No servers to move")
            return False

        # 随机选择一个服务器进行迁移
        server = random.choice(list(self.context.solution.server_map.values()))
        server_id = server.server_id
        server_copy = self.context.solution.get_server_copy(server_id)

        # self._print(f"Moving server {server_copy}")

        # 检查服务器是否已经达到最大寿命
        purchase_time = server_copy.buy_and_move_info[0].time_step
        max_dismiss_time = purchase_time + server_copy.life_expectancy
        max_dismiss_time = min(max_dismiss_time, TIME_STEPS)
        if server_copy.dismiss_time >= max_dismiss_time:
            self._print(f"Server {server_id} has reached maximum lifespan, cannot move")
            return False

        # 计算可迁移的时间步，迁移时间不能超过服务器的最大寿命
        earliest_move_time = server_copy.dismiss_time  # 在当前的 dismiss_time 进行迁移
        remaining_lifespan = max_dismiss_time - earliest_move_time
        if remaining_lifespan < 1:
            self._print(f"Not enough lifespan to move server {server_id}")
            return False

        # 随机决定新的 dismiss_time，在剩余寿命内
        additional_life = random.randint(1, remaining_lifespan)
        new_dismiss_time = earliest_move_time + additional_life

        self._print(f"Earliest move time: {earliest_move_time}, New dismiss time: {new_dismiss_time}, Remaining lifespan: {remaining_lifespan}, Max dismiss time: {max_dismiss_time}, Additional life: {additional_life}")

        # 随机选择一个不同的数据中心
        possible_datacenters = list(self.context.slot_manager.total_slots.keys())
        # 获取服务器当前所在的数据中心
        current_data_center = server_copy.buy_and_move_info[-1].target_datacenter
        possible_datacenters.remove(current_data_center)
        if not possible_datacenters:
            self._print(f"No other data centers to move server {server_id}")
            return False
        new_data_center = random.choice(possible_datacenters)

        slots_needed = server_copy.quantity * server_copy.slots_size

        # 检查新数据中心在迁移后的时间范围内是否有足够的插槽
        if not self.context.slot_manager.check_availability(earliest_move_time, new_dismiss_time, new_data_center, slots_needed):
            self._print(f"Not enough slots in {new_data_center} to move server {server_id}")
            return False

        # 更新 buy_and_move_info，添加新的迁移信息
        new_move_info = ServerMoveInfo(time_step=earliest_move_time, target_datacenter=new_data_center)
        server_copy.buy_and_move_info.append(new_move_info)
        server_copy.init_buy_and_move_info()

        # 更新 dismiss_time
        server_copy.dismiss_time = new_dismiss_time

        # 更新槽位管理器：释放旧数据中心的插槽，预订新数据中心的插槽
        old_data_center = current_data_center
        # self.context.slot_manager.push_slot_update(earliest_move_time, new_dismiss_time, old_data_center, slots_needed, 'cancel')
        self.context.slot_manager.push_slot_update(earliest_move_time, new_dismiss_time, new_data_center, slots_needed, 'buy')

        # 应用服务器变更
        self.context.solution.apply_server_change(server_copy)

        self._print(f"Moved server {server_id} from {old_data_center} to {new_data_center} at time {earliest_move_time}, new dismiss_time is {new_dismiss_time}")
        return True

class AdjustQuantityOperation(NeighborhoodOperation):
    MAX_QUANTITY_CHANGE = 0.12
    def execute(self):
        if not self.context.solution.server_map:
            self._print("No servers to adjust quantity")
            return False

        # 随机选择一个服务器
        server = random.choice(list(self.context.solution.server_map.values()))
        server_id = server.server_id
        current_quantity = server.quantity
        slots_size = server.slots_size

        # 决定增加还是减少
        if random.random() < 0.5:
            # 随机选择一个新的数量，介于 0 和 (current_quantity - 1) 之间
            new_quantity = random.randint(0, current_quantity - 1)
            quantity_change = current_quantity - new_quantity  # Positive number

            server_copy = copy.deepcopy(server)
            server_copy.quantity = new_quantity

            # 更新槽位管理器：在服务器存在的所有数据中心和时间段内释放相应的插槽
            for i, move_info in enumerate(server.buy_and_move_info):
                start_time = move_info.time_step
                if i + 1 < len(server.buy_and_move_info):
                    end_time = server.buy_and_move_info[i + 1].time_step
                else:
                    end_time = server.dismiss_time
                data_center = move_info.target_datacenter

                slots_needed = quantity_change * slots_size
                self.context.slot_manager.push_slot_update(start_time, end_time, data_center, slots_needed, 'cancel')  # 释放插槽

            # 应用服务器变更
            self.context.solution.apply_server_change(server_copy)

            self._print(f"Decreased quantity of server {server_id} from {current_quantity} to {new_quantity}")
            return True
        else:
            # 增加数量
            # 确定在服务器生命周期内所有数据中心和时间段内可增加的最大数量
            max_additional_quantity = None

            # 存储每个数据中心和时间段内需要的插槽数量
            slots_needed_per_period = []

            # 计算每个数据中心和时间段的可用插槽，确定可增加的最大数量
            for i, move_info in enumerate(server.buy_and_move_info):
                start_time = move_info.time_step
                if i + 1 < len(server.buy_and_move_info):
                    end_time = server.buy_and_move_info[i + 1].time_step
                else:
                    end_time = server.dismiss_time
                data_center = move_info.target_datacenter

                # 获取在该时间段和数据中心的最大可用插槽
                available_slots = self.context.slot_manager.get_maximum_available_slots(start_time, end_time, data_center)
                # 计算该时间段可增加的最大数量
                max_quantity = int(available_slots // slots_size * self.MAX_QUANTITY_CHANGE)

                if max_additional_quantity is None or max_quantity < max_additional_quantity:
                    max_additional_quantity = max_quantity

                slots_needed_per_period.append((start_time, end_time, data_center, available_slots))

            if max_additional_quantity <= 0:
                self._print(f"Not enough slots to increase quantity of server {server_id}")
                return False

            # 随机选择要增加的数量
            additional_quantity = random.randint(1, max_additional_quantity)
            new_quantity = current_quantity + additional_quantity

            server_copy = copy.deepcopy(server)
            server_copy.quantity = new_quantity

            # 检查并更新槽位管理器：在所有相关数据中心和时间段内预订插槽
            for start_time, end_time, data_center, _ in slots_needed_per_period:
                slots_needed = additional_quantity * slots_size
                # 检查插槽可用性
                if not self.context.slot_manager.check_availability(start_time, end_time, data_center, slots_needed):
                    self._print(f"Not enough slots in data center {data_center} from time {start_time} to {end_time} for server {server_id}")
                    return False
                # 预订插槽
                self.context.slot_manager.push_slot_update(start_time, end_time, data_center, slots_needed, 'buy')

            # 应用服务器变更
            self.context.solution.apply_server_change(server_copy)

            self._print(f"Increased quantity of server {server_id} from {current_quantity} to {new_quantity}")
            return True

class AdjustTimeOperation(NeighborhoodOperation):
    def execute(self):
        if not self.context.solution.server_map:
            self._print("No servers to adjust time")
            return False

        server = random.choice(list(self.context.solution.server_map.values()))
        server_copy = self.context.solution.get_server_copy(server.server_id)

        # 随机决定调整哪种时间
        operation_type = random.choice(['buy', 'move', 'dismiss'])

        if operation_type == 'buy':
            earliest_time, latest_time = self.get_adjustable_time_range(server_copy, 'buy')
            if earliest_time is None or latest_time is None:
                self._print(f"No valid purchase times available for server {server_copy.server_id}")
                self._print(f"Earliest time: {earliest_time}, Latest time: {latest_time}")
                return False
            new_purchase_time = random.randint(earliest_time, latest_time)
            self._print(f'调整服务器 {server_copy.server_id} 的购买时间为 {new_purchase_time} 原购买时间为 {server_copy.buy_and_move_info[0].time_step}')
            return self.adjust_purchase_time(server_copy, server, new_purchase_time)

        elif operation_type == 'move':
            result = self.get_adjustable_time_range(server_copy, 'move')
            if result is None:
                self._print(f"No migrations to adjust for server {server_copy.server_id}")
                return False
            earliest_time, latest_time, move_idx = result
            if earliest_time is None or latest_time is None:
                self._print(f"No valid migration times available for server {server_copy.server_id}")
                return False
            new_migration_time = random.randint(earliest_time, latest_time)
            return self.adjust_migration_time(server_copy, server, move_idx, new_migration_time)

        elif operation_type == 'dismiss':
            earliest_time, latest_time = self.get_adjustable_time_range(server_copy, 'dismiss')
            if earliest_time is None or latest_time is None:
                self._print(f"No valid dismiss times available for server {server_copy.server_id}")
                return False
            new_dismiss_time = random.randint(earliest_time, latest_time)
            return self.adjust_dismiss_time(server_copy, server, new_dismiss_time)

        else:
            return False

    def get_adjustable_time_range(self, server: ServerInfo, operation_type: str):
        # 获取服务器的属性
        purchase_time = server.buy_and_move_info[0].time_step
        max_life_expectancy = server.life_expectancy
        max_dismiss_time = min(purchase_time + max_life_expectancy, TIME_STEPS)
        next_move_time = server.buy_and_move_info[1].time_step if len(server.buy_and_move_info) > 1 else server.dismiss_time

        if operation_type == 'buy':
            # 获取服务器的发布窗口
            self._print(f'Server Info: {server}')
            server_info = self.context.servers_df[self.context.servers_df['server_generation'] == server.server_generation].iloc[0]
            release_start = server_info['release_start'] - 1  # 时间步索引从0开始
            release_end = server_info['release_end'] - 1
            dismiss_time = server.dismiss_time
            earliest_time_by_dismiss = dismiss_time - server.life_expectancy
            if earliest_time_by_dismiss < 0:
                earliest_time_by_dismiss = 0

            # 可调整的购买时间范围
            min_time = max(release_start, earliest_time_by_dismiss)
            max_time = min(release_end, server.dismiss_time - 1, max_dismiss_time - 1, next_move_time - 1)
            current_time = server.buy_and_move_info[0].time_step

            # 数据中心
            data_center = server.buy_and_move_info[0].target_datacenter

            earliest_time = self.context.slot_manager.find_time_step(
                data_center, 
                server.slots_size * server.quantity, 
                min_time, 
                current_time - 1, # 向前找能调整的位置
                sign=-1)
            
            self._print(f'Earliest time: {earliest_time}, Latest time: {max_time}')
            
            return earliest_time, max_time
            
        elif operation_type == 'move':
            if len(server.buy_and_move_info) <= 1:
                return None  # 无迁移可调整
            # 随机选择一个迁移
            move_indices = list(range(1, len(server.buy_and_move_info)))
            move_idx = random.choice(move_indices)
            current_time = server.buy_and_move_info[move_idx].time_step

            previous_time = server.buy_and_move_info[move_idx - 1].time_step + 1
            if move_idx + 1 < len(server.buy_and_move_info):
                next_time = server.buy_and_move_info[move_idx + 1].time_step - 1
            else:
                next_time = min(server.dismiss_time - 1, max_dismiss_time - 1)

            min_time = previous_time
            max_time = next_time

            data_center_curr = server.buy_and_move_info[move_idx].target_datacenter
            data_center_prev = server.buy_and_move_info[move_idx - 1].target_datacenter

            # 从当前位置向前 找到该数据中心 最早能够迁移的时间步骤 该步骤满足插槽需求 传入数据中心 要检查的时间范围 检查的方向 向前 需要的插槽数量
            # 从当前位置向后 找到该数据中心 最晚能够迁移的时间步骤 该步骤满足插槽需求 传入数据中心 要检查的时间范围 检查的方向 向前 需要的插槽数量
            # 这里需要双向查找
            #【在这里补充代码】
            earliest_time = self.context.slot_manager.find_time_step(
                data_center_curr,
                server.slots_size * server.quantity,
                min_time,
                current_time - 1, # 向前找能调整的位置
                sign=-1)
            # 注意这里是在调整前者时间长度
            latest_time = self.context.slot_manager.find_time_step(
                data_center_prev,
                server.slots_size * server.quantity,
                current_time, # 向后找能调整的位置
                max_time - 1,
                sign=1)
            latest_prev_time = latest_time + 1 if latest_time is not None else None
            return earliest_time, latest_prev_time, move_idx

        elif operation_type == 'dismiss':
            last_move_time = server.buy_and_move_info[-1].time_step
            min_time = last_move_time + 1
            max_time = max_dismiss_time
            current_time = server.dismiss_time

            data_center = server.buy_and_move_info[-1].target_datacenter
            # 注意这里调整的是前者时间长度
            latest_time = self.context.slot_manager.find_time_step(
                data_center,
                server.slots_size * server.quantity,
                current_time,   # 向后找能调整的位置
                max_time -1,
                sign=1)
            
            latest_dismiss_time = latest_time + 1 if latest_time is not None else None
            
            return min_time, latest_dismiss_time
            
        else:
            raise ValueError(f"Invalid operation type: {operation_type}")

    def adjust_purchase_time(self, server_copy:ServerInfo, original_server:ServerInfo, new_purchase_time):
        self._print(f"Adjusting buy time for server {server_copy.server_id}")
        server_copy.buy_and_move_info[0].time_step = new_purchase_time
        # 确保迁移时间序列严格递增
        if DEBUG and not self.check_time_sequence(server_copy):
            raise("Migration times are not strictly increasing after adjustment")
        return self.apply_time_adjustment(server_copy, original_server)

    def adjust_migration_time(self, server_copy:ServerInfo, original_server, move_idx:int, new_move_time):
        self._print(f"Adjusting move time for server {server_copy.server_id}")
        server_copy.buy_and_move_info[move_idx].time_step = new_move_time
        # 确保迁移时间序列严格递增
        if not self.check_time_sequence(server_copy):
            raise("Migration times are not strictly increasing after adjustment")
        return self.apply_time_adjustment(server_copy, original_server)

    def adjust_dismiss_time(self, server_copy:ServerInfo, original_server, new_dismiss_time):
        self._print(f"Adjusting dismiss time for server {server_copy.server_id}")
        server_copy.dismiss_time = new_dismiss_time
        if not self.check_time_sequence(server_copy):
            raise("Migration times are not strictly increasing after adjustment")
        return self.apply_time_adjustment(server_copy, original_server)

    def apply_time_adjustment(self, server_copy:ServerInfo, original_server):
        slots_needed = server_copy.quantity * server_copy.slots_size

        # 释放原先的插槽预订
        self.update_slots(original_server, slots_needed, 'cancel')

        # 预订新的插槽
        if not self.update_slots(server_copy, slots_needed, 'buy'):
            # 如果预订失败，恢复原先的插槽预订
            self.update_slots(original_server, slots_needed, 'buy')
            return False

        # 应用服务器变更
        self.context.solution.apply_server_change(server_copy)
        self._print(f"Adjusted time for server {server_copy.server_id}")
        return True

    def update_slots(self, server:ServerInfo, slots_needed, operation):
        # 遍历服务器的所有时间段，更新插槽
        for i, move_info in enumerate(server.buy_and_move_info):
            start_time = move_info.time_step
            if i + 1 < len(server.buy_and_move_info):
                end_time = server.buy_and_move_info[i + 1].time_step
            else:
                end_time = server.dismiss_time
            data_center = move_info.target_datacenter

            self.context.slot_manager.push_slot_update(start_time, end_time, data_center, slots_needed, operation)
        return True
    
    def check_time_sequence(self, server_copy:ServerInfo):
        times = [move_info.time_step for move_info in server_copy.buy_and_move_info]
        for earlier, later in zip(times, times[1:]):
            if earlier >= later:
                self._print("Migration times are not strictly increasing after adjustment")
                return False
        if server_copy.buy_and_move_info[0].time_step + server_copy.life_expectancy < server_copy.dismiss_time:
            self._print("Dismiss time is earlier than expected")
            return False
        return True

class SimulatedAnnealing:
    def __init__(self, slot_manager, servers_df, id_gen, solution: DiffSolution, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.id_gen = id_gen
        self.slot_manager: SlotAvailabilityManager = slot_manager
        self.verbose = verbose
        self.seed = seed
        self.solution = solution

        # 初始化操作上下文
        self.context = OperationContext(
            slot_manager=self.slot_manager,
            servers_df=servers_df,
            id_gen=self.id_gen,
            solution=self.solution,
            verbose=self.verbose
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

        self.best_solution = copy.deepcopy(self.solution)
        self.best_score = float('-inf')

    def _print(self, *args, color=None, **kwargs):
        if self.verbose:
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
        success = operation.execute()
        return success

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
            result = self.slot_manager.can_accommodate_servers(self.solution.server_map)
            if not result:
                self._print("New solution is invalid", color=Fore.RED)
                raise ValueError("New solution is invalid")
            self.slot_manager.apply_pending_updates()
            if new_score > self.best_score:
                self.best_solution = copy.deepcopy(self.solution)
                self.best_score = new_score
                self._print(f"New best solution with score {self.best_score:.5e}", color=Fore.GREEN)
            return True
        else:
            # 拒绝新解并回滚更改
            self.solution.discard_server_changes()
            self._print(f"Rejected new solution with score {new_score:.5e}", color=Fore.RED)
            return False

    def run(self):
        """模拟退火的主循环。"""
        current_score = self.solution.diff_evaluation()  # 初始评价
        iteration = 0  # 用于记录有效迭代次数
        while iteration < self.max_iter:
            self._print(f"<------ Iteration {iteration}, Temperature {self.current_temp:.2f} ------>")
            
            if self.generate_neighbor():  # 生成一个邻域解
                new_score = self.solution.diff_evaluation()  # 评估新解
                accept_prob = self.acceptance_probability(current_score, new_score)
                if self.verbose == False:
                    print(f"Iteration: {iteration}. New best solution with score {self.best_score:.5e}")
                if self.accept_solution(accept_prob, new_score):
                    current_score = new_score  # 如果接受，更新当前分数
                    
                # 只有当找到有效邻域解时，才增加迭代次数
                iteration += 1
                self.current_temp *= self.alpha  # 降低温度
                if self.current_temp < self.min_temp:
                    break
            else:
                self._print("No valid neighbor found")

        return self.best_solution, self.best_score

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
        alpha=0.9998,
        max_iter=30000,
        verbose=verbose
    )
    best_solution, best_score = sa.run()
    print(f'Final best score for {seed}: {best_score:.5e}')
    return best_solution, best_score

if __name__ == '__main__':
    start = time.time()
    seed = 3329
    best_solution, best_score = get_my_solution(seed, verbose=False)
    best_solution.export_solution_to_json(f"./output/{seed}_{best_score:.5e}.json")
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")
