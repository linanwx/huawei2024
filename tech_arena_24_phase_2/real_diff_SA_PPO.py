import ast
import math
import os
import time
import copy
import random
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from colorama import Fore, Style

# Import the new DiffSolution and related classes
from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo, export_solution_to_json, update_best_solution, SERVER_GENERATION_MAP, LATENCY_SENSITIVITY_MAP, evaluate_map
from idgen import ThreadSafeIDGenerator

# from ppo_sa_env import PPO_SA_Env

from real_diff_SA_basic import NeighborhoodOperation, SA_status, SlotAvailabilityManager, OperationContext

TIME_STEPS = 168
DEBUG = False

# INITIAL_TEMPERATURE = 897000000.0
# MIN_TEMPERATURE = 0.000001
# ALPHA = 0.9999656
# MAX_ITER = 1000000

INITIAL_TEMPERATURE = 4480000.0
MIN_TEMPERATURE = 44.8
ALPHA = 0.9999885
MAX_ITER = 1000000
GLOBAL_MAX_PURCHASE_RATIO = 0.12

# Automatically create output directory
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    MAX_PURCHASE_RATIO = GLOBAL_MAX_PURCHASE_RATIO
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
            self._print(f"Server {server_id} has reached maximum lifespan, from {purchase_time} to {server_copy.dismiss_time}, life {server_copy.dismiss_time - purchase_time}, max life {max_dismiss_time - purchase_time}", color=Fore.YELLOW)
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

class AdjustQuantityOperation(SA_NeighborhoodOperation):
    MAX_QUANTITY_CHANGE = GLOBAL_MAX_PURCHASE_RATIO
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
            max_quantity_list = []

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
                if available_slots // slots_size >= 1 and max_quantity == 0:
                    max_quantity = 1 # 确保至少增加1个
                max_quantity_list.append(max_quantity)

                if max_additional_quantity is None or max_quantity < max_additional_quantity:
                    max_additional_quantity = max_quantity

                slots_needed_per_period.append((start_time, end_time, data_center, available_slots))

            if max_additional_quantity <= 0:
                self._print(f"Not enough slots to increase quantity of server {server_id}")
                self._print(f"Max additional quantity: {max_additional_quantity}, Slots needed per period: {slots_needed_per_period}, Max quantity list: {max_quantity_list}")
                self._print(f"Server: {server}")
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

class AdjustTimeOperation(SA_NeighborhoodOperation):
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
            self._print(f'Adjustable time range for buy operation: {earliest_time} - {latest_time}')
            if earliest_time is None or latest_time is None:
                self._print(f"No valid purchase times available for server {server_copy.server_id}")
                self._print(f"Earliest time: {earliest_time}, Latest time: {latest_time}")
                return False
            new_purchase_time = random.randint(earliest_time, latest_time)
            self._print(f'调整服务器 {server_copy.server_id} 的购买时间为 {new_purchase_time} 原购买时间为 {server_copy.buy_and_move_info[0].time_step}')
            return self.adjust_purchase_time(server_copy, server, new_purchase_time)

        elif operation_type == 'move':
            result = self.get_adjustable_time_range(server_copy, 'move')
            self._print(f'Adjustable time range for move operation: {result}')
            if result is None:
                self._print(f"No migrations to adjust for server {server_copy.server_id}", color=Fore.YELLOW)
                return False
            earliest_time, latest_time, move_idx = result
            if earliest_time is None or latest_time is None:
                self._print(f"No valid migration times available for server {server_copy.server_id}")
                return False
            new_migration_time = random.randint(earliest_time, latest_time)
            return self.adjust_migration_time(server_copy, server, move_idx, new_migration_time)

        elif operation_type == 'dismiss':
            self._print(f"Adjusting dismiss time for server {server_copy.server_id}, current dismiss time: {server_copy.dismiss_time}, purchase time: {server_copy.buy_and_move_info[0].time_step}")
            earliest_time, latest_time = self.get_adjustable_time_range(server_copy, 'dismiss')
            self._print(f'Adjustable time range for dismiss operation: {earliest_time} - {latest_time}')
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
            self._print(f'Release start: {release_start}, Release end: {release_end}')
            dismiss_time = server.dismiss_time
            earliest_time_by_dismiss = dismiss_time - server.life_expectancy
            if earliest_time_by_dismiss < 0:
                earliest_time_by_dismiss = 0
            self._print(f'Earliest time by dismiss: {earliest_time_by_dismiss}, Dismiss time: {dismiss_time}')

            # 可调整的购买时间范围
            min_time = max(release_start, earliest_time_by_dismiss)
            max_time = min(release_end, server.dismiss_time - 1, max_dismiss_time - 1, next_move_time - 1)
            current_time = server.buy_and_move_info[0].time_step
            self._print(f'Earliest time: {min_time}, Latest time: {max_time}, Current time: {current_time}')

            # 数据中心
            data_center = server.buy_and_move_info[0].target_datacenter

            earliest_time = self.context.slot_manager.find_time_step(
                data_center, 
                server.slots_size * server.quantity, 
                min_time, 
                current_time - 1, # 向前找能调整的位置
                sign=-1)
            
            if earliest_time is None:
                earliest_time = current_time
            
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
            if earliest_time is None:
                earliest_time = current_time
            # 注意这里是在调整前者时间长度
            latest_time = self.context.slot_manager.find_time_step(
                data_center_prev,
                server.slots_size * server.quantity,
                current_time, # 向后找能调整的位置
                max_time - 1,
                sign=1)
            latest_prev_time = latest_time + 1 if latest_time is not None else current_time
            return earliest_time, latest_prev_time, move_idx

        elif operation_type == 'dismiss':
            last_move_time = server.buy_and_move_info[-1].time_step
            min_time = last_move_time + 1
            max_time = max_dismiss_time
            current_time = server.dismiss_time
            data_center = server.buy_and_move_info[-1].target_datacenter
            self._print(f'Last move time: {last_move_time}, Min dismiss time: {min_time}, Max dismiss time: {max_time}, Current dismiss time: {current_time}')

            # 注意这里调整的是前者时间长度
            latest_time = self.context.slot_manager.find_time_step(
                data_center,
                server.slots_size * server.quantity,
                current_time,   # 向后找能调整的位置
                max_time -1,
                sign=1)
            
            latest_dismiss_time = latest_time + 1 if latest_time is not None else current_time
            
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
        self.push_slot_update(original_server, slots_needed, 'cancel')

        # 预订新的插槽
        if not self.push_slot_update(server_copy, slots_needed, 'buy'):
            # 如果预订失败，恢复原先的插槽预订
            self.push_slot_update(original_server, slots_needed, 'buy')
            return False

        # 应用服务器变更
        self.context.solution.apply_server_change(server_copy)
        self._print(f"Adjusted time for server {server_copy.server_id}")
        return True

    def push_slot_update(self, server:ServerInfo, slots_needed, operation):
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
    
class RemoveServerOperation(SA_NeighborhoodOperation):
    def execute(self):
        if not self.context.solution.server_map:
            self._print("No servers to remove")
            return False

        # Decide whether to delete a server or delete a move record
        if random.random() < 0.5:
            return self.delete_server()
        else:
            return self.delete_move_record()

    def delete_server(self):
        # Randomly select a server to delete
        server = random.choice(list(self.context.solution.server_map.values()))
        server_id = server.server_id
        self._print(f"Deleting server {server_id}")

        server_copy = self.context.solution.get_server_copy(server_id)

        # Store slots size and compute total slots to release
        slots_size = server_copy.slots_size
        total_slots_to_cancel = server_copy.quantity * slots_size

        # Push slot updates to release slots for all periods and data centers
        for i, move_info in enumerate(server_copy.buy_and_move_info):
            start_time = move_info.time_step
            if i + 1 < len(server_copy.buy_and_move_info):
                end_time = server_copy.buy_and_move_info[i + 1].time_step
            else:
                end_time = server_copy.dismiss_time
            data_center = move_info.target_datacenter

            self.context.slot_manager.push_slot_update(
                start_time, end_time, data_center, total_slots_to_cancel, 'cancel'
            )

        # Remove the server by setting the quantity to zero
        server_copy.quantity = 0

        # Apply server change
        self.context.solution.apply_server_change(server_copy)

        return True

    def delete_move_record(self):
        # Collect servers with more than one move record
        servers_with_moves = [
            s for s in self.context.solution.server_map.values()
            if len(s.buy_and_move_info) > 1
        ]
        if not servers_with_moves:
            self._print("No servers with move records to delete")
            return False

        # Randomly select a server and a move record to delete (excluding the initial buy)
        server = random.choice(servers_with_moves)
        server_id = server.server_id
        server_copy = self.context.solution.get_server_copy(server_id)

        move_indices = list(range(1, len(server_copy.buy_and_move_info)))
        move_idx_to_delete = random.choice(move_indices)
        move_info_to_delete = server_copy.buy_and_move_info[move_idx_to_delete]

        self._print(f"Deleting move record at index {move_idx_to_delete} from server {server_id}")

        # Store slots size and compute total slots to adjust
        slots_size = server_copy.slots_size
        slots_needed = server_copy.quantity * slots_size

        # Determine time range of the move to delete
        start_time = move_info_to_delete.time_step
        if move_idx_to_delete + 1 < len(server_copy.buy_and_move_info):
            end_time = server_copy.buy_and_move_info[move_idx_to_delete + 1].time_step
        else:
            end_time = server_copy.dismiss_time

        data_center = move_info_to_delete.target_datacenter

        # Remove the move record
        del server_copy.buy_and_move_info[move_idx_to_delete]

        # Ensure move times are strictly increasing
        if not self.check_time_sequence(server_copy):
            self._print("Move times are not strictly increasing after deletion")
            return False

        # Now, the previous move extends to the end time
        prev_move_info = server_copy.buy_and_move_info[move_idx_to_delete - 1]
        prev_data_center = prev_move_info.target_datacenter

        # The extended period is from the start_time of the deleted move to end_time
        extended_start_time = start_time
        extended_end_time = end_time

        # Check if the previous data center can accommodate the server during the extended period
        if not self.context.slot_manager.check_availability(
            extended_start_time, extended_end_time, prev_data_center, slots_needed
        ):
            self._print(
                f"Not enough slots in data center {prev_data_center} from time {extended_start_time} to {extended_end_time} "
                f"to accommodate server {server_id} after deleting move record", color=Fore.YELLOW
            )
            return False

        # Push slot updates
        # Release slots in the data center of the deleted move info
        self.context.slot_manager.push_slot_update(
            extended_start_time, extended_end_time, data_center, slots_needed, 'cancel'
        )

        # Reserve slots in the previous data center for the extended period
        self.context.slot_manager.push_slot_update(
            extended_start_time, extended_end_time, prev_data_center, slots_needed, 'buy'
        )

        # Apply server change
        self.context.solution.apply_server_change(server_copy)

        self._print(
            f"Deleted move record at index {move_idx_to_delete} from server {server_id}. "
            f"Server now stays at data center {prev_data_center} from time {prev_move_info.time_step} to {extended_end_time}"
        )

        return True

    def check_time_sequence(self, server_copy: ServerInfo):
        times = [move_info.time_step for move_info in server_copy.buy_and_move_info]
        for earlier, later in zip(times, times[1:]):
            if earlier >= later:
                self._print("Move times are not strictly increasing after deletion")
                return False
        return True
    
class MergeServersOperation(SA_NeighborhoodOperation):
    def execute(self):
        # Need at least two servers to attempt merging
        server_map = self.context.solution.server_map
        if len(server_map) < 2:
            self._print("Not enough servers to perform merge operation")
            return False

        # Group servers by server_generation to find pairs of the same type
        servers_by_generation = {}
        for server in server_map.values():
            gen = server.server_generation
            servers_by_generation.setdefault(gen, []).append(server)

        # Collect potential pairs for merging
        potential_pairs = []
        for servers in servers_by_generation.values():
            if len(servers) < 2:
                continue
            # Consider all pairs of servers of the same type
            for i in range(len(servers)):
                for j in range(i+1, len(servers)):
                    s1, s2 = servers[i], servers[j]
                    if s1.dismiss_time >= s2.buy_and_move_info[0].time_step:
                        continue
                    # Check if s1's max dismiss time is after s2's dismiss time
                    max_dismiss_time_s1 = min(s1.buy_and_move_info[0].time_step + s1.life_expectancy, TIME_STEPS)
                    if max_dismiss_time_s1 < s2.dismiss_time:
                        continue
                    potential_pairs.append((s1, s2))

        if not potential_pairs:
            self._print("No suitable server pairs found for merging")
            return False

        # Randomly select a pair to attempt merging
        s1, s2 = random.choice(potential_pairs)
        self._print(f"Attempting to merge Server {s1.server_id} and Server {s2.server_id}")
        self._print(f"Server 1: {s1}")
        self._print(f"Server 2: {s2}")

        # Begin merging process
        try:
            return self.merge_servers(s1, s2)
        except Exception as e:
            self._print(f"Error during merge: {e}", color=Fore.RED)
            return False

    def merge_servers(self, s1: ServerInfo, s2: ServerInfo):
        # Suppose the first server's dismiss time is t1, the second server's purchase time is t2
        t1 = s1.dismiss_time
        t2 = s2.buy_and_move_info[0].time_step

        # Check if there are enough slots between t1 and t2
        last_move_info_s1 = s1.buy_and_move_info[-1]
        data_center1 = last_move_info_s1.target_datacenter
        slots_needed = min(s1.quantity * s1.slots_size, s2.quantity * s2.slots_size)

        if not self.context.slot_manager.check_availability(t1, t2, data_center1, slots_needed):
            self._print(f"Not enough slots in data center {data_center1} between {t1} and {t2} for merging")
            return False

        # Copy server 1 into s1_new
        s1_new = copy.deepcopy(s1)
        s2_new = copy.deepcopy(s2)
        s1_new.quantity = min(s1.quantity, s2.quantity)

        # Extend s1_new's dismiss_time to s2's dismiss_time or its maximum life expectancy
        max_dismiss_time_s1 = min(s1.buy_and_move_info[0].time_step + s1.life_expectancy, TIME_STEPS)
        desired_dismiss_time = min(s2.dismiss_time, max_dismiss_time_s1)
        s1_new.dismiss_time = desired_dismiss_time

        # Transfer s2's purchase and migration records into s1_new
        first_move_info_s2 = s2.buy_and_move_info[0]
        data_center2 = first_move_info_s2.target_datacenter

        s2_buy_and_move_info = s2.buy_and_move_info.copy()
        if data_center1 == data_center2:
            # If the last move of s1_new and the first move of s2 are in the same data center, merge them
            # Remove the last move_info from s1_new
            s2_buy_and_move_info.pop()
        # Append s2's buy_and_move_info to s1_new
        s1_new.buy_and_move_info.extend(s2_buy_and_move_info)

        # Re-initialize cumulative durations
        s1_new.init_buy_and_move_info()

        # Ensure move times are strictly increasing
        if not self.check_time_sequence(s1_new):
            self._print("Move times are not strictly increasing after merging")
            return False

        # Copy server 2 into s2_new and set its quantity to 0
        s2_new.quantity = 0
        
        # Apply time adjustments
        self._print(f'Server 1 New: {s1_new}')
        self._print(f'Server 2 New: {s2_new}')

        if not self.apply_time_adjustment(s1_new, s1):
            self._print("Failed to apply time adjustment to merged server")
            return False
        if not self.apply_time_adjustment(s2_new, s2):
            self._print("Failed to apply time adjustment to server being removed")
            return False
        self._print(f'self.context.solution.server_map.length: {len(self.context.solution.server_map)}')
        self._print(f"Successfully merged server {s2.server_id} into server {s1_new.server_id}")
        return True
    
    def apply_time_adjustment(self, server_copy:ServerInfo, original_server:ServerInfo):
        # slots_needed = server_copy.quantity * server_copy.slots_size

        # 释放原先的插槽预订
        self.push_slot_update(original_server, original_server.quantity * original_server.slots_size , 'cancel')

        # 预订新的插槽
        if not self.push_slot_update(server_copy, server_copy.quantity * server_copy.slots_size, 'buy'):
            return False

        # 应用服务器变更
        self.context.solution.apply_server_change(server_copy)
        return True

    def push_slot_update(self, server:ServerInfo, slots_needed, operation):
        if slots_needed == 0:
            return True
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

    def check_time_sequence(self, server: ServerInfo) -> bool:
        """
        Ensures that move times in server's buy_and_move_info are strictly increasing.
        """
        times = [move_info.time_step for move_info in server.buy_and_move_info]
        if not all(earlier < later for earlier, later in zip(times, times[1:])):
            self._print("Move times are not strictly increasing")
            return False
        if server.dismiss_time <= times[-1]:
            self._print("Dismiss time is not after last move time")
            return False
        return True

class AdjustServerPriceOperation(SA_NeighborhoodOperation):
    LATENCY_SENSITIVITY_KEYS = list(LATENCY_SENSITIVITY_MAP.keys())
    SERVER_TYPE_KEYS = list(SERVER_GENERATION_MAP.keys())
    

    def execute(self):
        # 随机生成区间长度
        duration = random.randint(1, TIME_STEPS - 1)
        
        # 再随机生成起始时间，保证剩下的时间足够
        start_time = random.randint(0, TIME_STEPS - duration)
        
        # 根据区间长度确定结束时间
        end_time = start_time + duration
        
        latency_sensitivity = random.choice(self.LATENCY_SENSITIVITY_KEYS)
        server_type = random.choice(self.SERVER_TYPE_KEYS)
        ratio = random.uniform(0.5, 1.5)

        try:
            self.context.solution.adjust_price_ratio(start_time, end_time, latency_sensitivity,
                                                    server_type, ratio)
            self._print(f"Adjusted price ratio for server type {server_type} with latency sensitivity {latency_sensitivity} from {start_time} to {end_time} to {ratio}")
            return True
        except:
            return False

def calculate_steps(T0, T_final, alpha):
    if T0 <= T_final:
        raise ValueError("初始温度必须大于最终温度")
    if not (0 < alpha < 1):
        raise ValueError("降温系数 alpha 必须在 0 到 1 之间")

    steps = math.log(T_final / T0) / math.log(alpha)
    return math.ceil(steps)

class SimulatedAnnealing:
    def __init__(self, slot_manager, servers_df, id_gen, solution: DiffSolution, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        # 基于温度 重算最大步骤数
        self.status = SA_status()
        self.status.max_iter_by_temp = calculate_steps(initial_temp, min_temp, alpha)
        self.status.max_iter = min(max_iter, self.status.max_iter_by_temp)
        self.status.current_temp = initial_temp
        self.status.min_temp = min_temp
        self.status.alpha = alpha
        self.status.verbose = verbose
        self.status.seed = seed
        self.id_gen = id_gen

        self.slot_manager: SlotAvailabilityManager = slot_manager

        # 初始化操作上下文
        self.context = OperationContext(
            slot_manager=self.slot_manager,
            servers_df=servers_df,
            id_gen=self.id_gen,
            solution=solution,
            verbose=self.status.verbose,
            sa_status=self.status
        )

        # Initialize the PPO environment and model
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]
        )
        
        # self.env = PPO_SA_Env(solution, self.context, slot_manager, servers_df, id_gen)
        # self.model = PPO('MultiInputPolicy', self.env, verbose=1, policy_kwargs=policy_kwargs)
        # if os.path.exists(self.env.MODEL_SAVE_PATH):
        #     self.model.load(self.env.MODEL_SAVE_PATH)
        # self.env.model = self.model

        # 初始化操作
        self.operations : list[NeighborhoodOperation] = []
        self.operation_probabilities = []
        self.operation_enable_percentage = []
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
        self.register_operation(
            MergeServersOperation(context=self.context),
            weight=0.3  # Adjust the weight as desired
        )
        self.register_operation(
            AdjustServerPriceOperation(context=self.context),
            weight=0.6,
            enable_percentage=0.4
        )
        # self.register_operation(
        #     PPO_BuyServerOperation(context=self.context, env=self.env, model=self.model),
        #     weight=0.4
        # )

        self.best_solution_server_map = copy.deepcopy(self.context.solution.server_map)
        self.operation_record = {}

    def record_operation(self, operation: NeighborhoodOperation, success: bool):
        operation_name = operation.__class__.__name__
        if operation_name not in self.operation_record:
            self.operation_record[operation_name] = {'total': 0, 'success': 0}
        self.operation_record[operation_name]['total'] += 1
        if success:
            self.operation_record[operation_name]['success'] += 1

    def print_operation_record(self):
        self._print("Operation record:")
        for operation_name, record in self.operation_record.items():
            total = record['total']
            success = record['success']
            success_rate = success / total if total > 0 else 0
            self._print(f"{operation_name}: {success}/{total} ({success_rate:.2f})")

    def _print(self, *args, color=None, **kwargs):
        if self.status.verbose:
            # 设置颜色
            if color:
                print(f"{color}{' '.join(map(str, args))}{Style.RESET_ALL}", **kwargs)
            else:
                print(*args, **kwargs)

    def register_operation(self, operation, weight=1.0, enable_percentage=0.0):
        self.operations.append(operation)
        self.operation_probabilities.append(weight)
        self.total_weight += weight
        # 将启用百分比与操作一起存储
        self.operation_enable_percentage.append(enable_percentage)

    def choose_operation(self):
        """
        在选择操作之前检查其启用条件。
        只有达到启用百分比要求的操作才会被选择。
        """
        # 获取当前迭代数与最大迭代数
        current_step = self.status.iteration
        max_steps = self.status.max_iter
        current_percentage = current_step / max_steps
        # 过滤掉那些尚未启用的操作
        available_operations = [
            (op, weight) for op, weight, enable_percent in zip(self.operations, self.operation_probabilities, self.operation_enable_percentage)
            if current_percentage >= enable_percent
        ]
        if not available_operations:
            raise ValueError("No operations are available at this stage.")
        # 解包操作与权重列表
        operations, weights = zip(*available_operations)
        # 计算权重总和
        total_weight = sum(weights)
        # 计算权重比例
        probabilities = [w / total_weight for w in weights]
        # 随机选择一个操作
        return random.choices(operations, weights=probabilities, k=1)[0]

    def generate_neighbor(self):
        self.slot_manager.clear_pending_updates()
        operation = self.choose_operation()
        success, score  = operation.execute_and_evaluate()
        self._print(f"Operation: {operation.__class__.__name__}, Score: {score:.5e}, Success: {success}")
        self.record_operation(operation, success)
        return score, success, operation

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.status.current_temp)

    def accept_solution(self, accept_prob, new_score):
        if accept_prob >= 1.0 or random.random() < accept_prob:
            self._print(f"Accepted new solution with score {new_score:.5e}", color=Fore.BLUE)
            # 接受新解
            self.context.solution.commit_server_changes()
            # 检查解是否合法
            self.slot_manager.apply_pending_updates()
            if DEBUG:
                result = self.slot_manager.can_accommodate_servers(self.context.solution.server_map)
                if not result:
                    self._print("New solution is invalid", color=Fore.RED)
                    raise ValueError("New solution is invalid")
            if new_score > self.status.best_score:
                self.best_solution_server_map = update_best_solution(self.best_solution_server_map, self.context.solution.server_map)
                self.status.best_score = new_score
                self.status.best_price_matrix = self.context.solution.price_matrix.copy()
                self._print(f"New best solution with score {self.status.best_score:.5e}", color=Fore.GREEN)
            return True
        else:
            # 拒绝新解并回滚更改
            self.context.solution.discard_server_changes()
            self._print(f"Rejected new solution with score {new_score:.5e}")
            return False

    def run(self):
        """模拟退火的主循环。"""
        self.status.current_score = self.context.solution.diff_evaluation()  # 初始评价
        self.status.iteration = 0  # 用于记录有效迭代次数
        while self.status.iteration < self.status.max_iter:
            self._print(f"<------ Iteration {self.status.iteration}/Max:{self.status.max_iter}/Max by temp:{self.status.max_iter_by_temp}, Temperature {self.status.current_temp:.2f} Bestscore {self.status.best_score:.5e} ------->", color=Fore.CYAN)
            self.print_operation_record()
            new_score, success, _ = self.generate_neighbor()  # 生成一个邻域解
            if success:
                accept_prob = self.acceptance_probability(self.status.current_score, new_score)
                # print(f"Iteration: {iteration}. New best solution for {self.status.seed} with score {self.status.best_score:.5e}")
                if self.accept_solution(accept_prob, new_score):
                    self.status.current_score = new_score  # 如果接受，更新当前分数
                    # score_compaire, another_S = evaluate_map(self.context.sa_status.seed, self.context.solution.server_map)
                    # self.context.solution.check_same(another_S)
                    # print(f'score: {self.status.current_score} score_compaire:{score_compaire}')
                    # if math.fabs(score_compaire - self.status.current_score) > 1e-6:
                    #     raise("score_compaire != self.status.current_score")
                # 只有当找到有效邻域解时，才增加迭代次数
                self.status.iteration += 1
                self.status.current_temp *= self.status.alpha  # 降低温度
                if self.status.current_temp < self.status.min_temp:
                    break
            else:
                self._print("No valid neighbor found", color=Fore.RED)

        return self.best_solution_server_map, self.status.best_score, self.status.best_price_matrix

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
        initial_temp=INITIAL_TEMPERATURE,
        min_temp=MIN_TEMPERATURE,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        verbose=verbose
    )
    best_solution_server_map, best_score, best_price_matrix = sa.run()
    print(f'Final best score for {seed}: {best_score:.5e}')
    export_solution_to_json(best_solution_server_map, best_price_matrix, f"./output/{seed}_{best_score:.5e}.json")
    return best_solution_server_map, best_score

if __name__ == '__main__':
    start = time.time()
    seed = 2381
    best_solution_server_map, best_score = get_my_solution(seed, verbose=True if DEBUG else False)
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")
