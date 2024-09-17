from datetime import datetime
import math
import os
import time
import numpy as np
import pandas as pd
import random
import copy
import threading
import logging

from evaluation_diff_quantity import DiffInput, DiffSolution
from utils import save_solution

from real_diff_evaluation import DiffSolution as RealDiffSolution, ServerInfo, ServerMoveInfo

TIME_STEPS = 168
generate_new_time_step_STEP = 20
MAX_BUY_PERCENTAGE = 0.12
MAX_ADD_PERCENTAGE = 0.12

# 自动创建 output 目录
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成带有时间戳的日志文件名
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'{output_dir}simulation_{current_time}.log'

# 创建一个新的 logger，使用唯一名称，确保与其他 logger 独立
logger = logging.getLogger(f'custom_logger_{current_time}')
logger.setLevel(logging.DEBUG)  # 设置日志级别

# 避免当前 logger 传播到父级 logger（即不使用根 logger）
logger.propagate = False

# 如果 logger 没有 handler，添加新的 file handler
if not logger.hasHandlers():
    # 创建文件 handler
    file_handler = logging.FileHandler(log_filename)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将 handler 添加到 logger 中
    logger.addHandler(file_handler)

# 记录测试日志
logger.debug("This is a debug message.")
logger.info("Logger is configured correctly.")

def generate_new_time_step(old_time_step, TIME_STEPS):
    min_shift = max(-generate_new_time_step_STEP, 1 - old_time_step)  # 确保不小于1
    max_shift = min(generate_new_time_step_STEP, TIME_STEPS - old_time_step)  # 确保不大于TIME_STEPS

    if min_shift > max_shift:
        return None, "无法生成有效的新时间段"

    new_time_step = old_time_step + random.randint(min_shift, max_shift)
    return new_time_step, "新时间段生成成功"


class ThreadSafeIDGenerator:
    def __init__(self, start=0):
        self.current_id = start
        self.lock = threading.Lock()

    def next_id(self):
        with self.lock:
            self.current_id += 1
            return str(self.current_id)


class SlotAvailabilityManager:
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
            logger.debug(" ".join(map(str, args)))

    def __init__(self, datacenters, verbose=False):
        # 初始化 0-168 的插槽表，第 0 行将保持空置
        self.slots_table = pd.DataFrame({'time_step': np.arange(0, TIME_STEPS + 1)})
        self.verbose = verbose
        for dc in datacenters['datacenter_id']:
            # 从第 1 行到第 168 行初始化插槽容量
            self.slots_table[dc] = [0] + [datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0]] * TIME_STEPS
        self.total_slots = {dc: datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0] for dc in datacenters['datacenter_id']}

    def check_availability(self, start_time, end_time, data_center, slots_needed):
        # 直接使用 1 到 168 的时间步长
        end_time = min(end_time, TIME_STEPS)
        for ts in range(start_time, end_time + 1):
            if self.slots_table.at[ts, data_center] < slots_needed:
                return False
        return True

    def get_maximum_available_slots(self, start_time, end_time, data_center):
        # 检查时使用 0 到 168，但从第 1 行开始计算可用插槽
        end_time = min(end_time, TIME_STEPS + 1)
        available_slots = self.slots_table.loc[start_time:end_time - 1, data_center].min()  # 确保包含 start_time，不包含 end_time
        return available_slots

    def update_slots(self, start_time, end_time, data_center, slots_needed, operation='buy'):
        end_time = min(end_time, TIME_STEPS + 1)
        for ts in range(start_time, end_time):
            if operation == 'buy':
                self.slots_table.at[ts, data_center] -= slots_needed
            elif operation == 'cancel':
                self.slots_table.at[ts, data_center] += slots_needed
        if operation == 'buy':
            self._print(f"购买了 {slots_needed * (end_time - start_time)} 个插槽，时间段 {start_time} - {end_time}, 数据中心 {data_center}")
        elif operation == 'cancel':
            self._print(f"取消了 {slots_needed * (end_time - start_time)} 个插槽，时间段 {start_time} - {end_time}, 数据中心 {data_center}")

    def simulate_slot_availability(self, start_time, end_time, data_center, slots_needed, existing_start, existing_end):
        """
        模拟调整时间段后的插槽可用性检查，不实际更改插槽表。
        """
        end_time = min(end_time, TIME_STEPS)
        existing_end = min(existing_end, TIME_STEPS)

        for ts in range(start_time, end_time + 1):
            simulated_slots = self.slots_table.at[ts, data_center]
            
            # 如果时间段内是已经被占用的部分，则先加回原来占用的插槽
            if existing_start <= ts <= existing_end:
                simulated_slots += slots_needed

            # 检查新的时间段是否可行
            if simulated_slots < slots_needed:
                return False

        return True

    def check_slot_consistency(self, solution, servers):
        """
        检查插槽占用情况与 slots_table 的一致性，并确保插槽数量大于等于 0
        :param solution: 当前的解决方案
        :param servers: 服务器数据，包含插槽大小信息
        :return: 如果一致且没有负值插槽，返回 True；否则，返回 False 并打印错误信息
        """
        slot_usage = pd.DataFrame({'time_step': np.arange(0, TIME_STEPS + 1)})
        for dc in self.total_slots.keys():
            slot_usage[dc] = 0  # 初始化为 0 插槽占用
        
        # 遍历 solution 中的每一行，更新 slot_usage 的插槽使用情况
        for _, row in solution.iterrows():
            time_step = row['time_step']
            dismiss_life_expectancy = row['dismiss_life_expectancy']
            datacenter_id = row['datacenter_id']
            quantity = row['quantity']
            
            # 获取当前 server_generation 对应的 slots_size
            slots_size = servers.loc[servers['server_generation'] == row['server_generation'], 'slots_size'].values[0]

            # 更新从购买时间到结束时间内的插槽占用情况
            for ts in range(time_step, min(time_step + dismiss_life_expectancy, TIME_STEPS + 1)):
                slot_usage.at[ts, datacenter_id] += quantity * slots_size

        flag = True  # 一致性检查标志
        negative_slot_flag = False  # 是否有负值插槽标志

        # 对比 slot_usage 和 slots_table
        for ts in range(1, TIME_STEPS + 1):
            for dc in self.total_slots.keys():
                expected_usage = slot_usage.at[ts, dc]
                actual_usage = self.total_slots[dc] - self.slots_table.at[ts, dc]

                if expected_usage != actual_usage:
                    self._print(f"Inconsistency found at time_step {ts}, datacenter {dc}: "
                                f"Expected {expected_usage} slots used, "
                                f"but found {actual_usage} slots used.")
                    flag = False

                # 检查插槽是否为负值
                if self.slots_table.at[ts, dc] < 0:
                    self._print(f"Negative slot value at time_step {ts}, datacenter {dc}: "
                                f"Available slots: {self.slots_table.at[ts, dc]}.")
                    negative_slot_flag = True

        if flag and not negative_slot_flag:
            self._print("Slot usage is consistent with the solution and no negative slots found.")
            return True

        if negative_slot_flag:
            self._print("Negative slot values found, please check the slot allocation logic.")
        
        if not flag:
            self._print("Slot usage is inconsistent with the solution.")
        
        return False

class SimulatedAnnealing:
    def __init__(self, initial_solution, slot_manager:SlotAvailabilityManager, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        self.current_solution = initial_solution
        self.best_solution = copy.deepcopy(initial_solution)
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.id_gen = ThreadSafeIDGenerator(start=0)
        self.pending_updates = []  
        self.slot_manager = slot_manager
        self.verbose = verbose
        self.seed = seed
        self.S = RealDiffSolution(seed=seed)

        self.operation_probabilities = {
            'buy': 4,
            'adjust_dismiss_time': 1,
            'adjust_time_slot': 1,
            "adjust_quantity": 1
        }
        self.low_pass_filter_alpha = 0.99
        self.initial_operation_probabilities = copy.deepcopy(self.operation_probabilities)

        self.step_count = int(math.log(min_temp / initial_temp) / math.log(alpha))
        self._print(f"将在 {self.step_count} 步中完成退火过程")

        # self.operation_success_counts = {op: 0 for op in self.operation_probabilities.keys()}

        # 详细操作统计，细分操作类别
        self.detailed_operation_counts = {
            'buy_life_expectancy': 0,
            'buy_custom_life': 0,
        }

    def log_detailed_operation(self, operation_type, sub_type):
        """用于记录具体的操作细分类型"""
        detailed_key = f"{operation_type}_{sub_type}"
        if detailed_key in self.detailed_operation_counts:
            self.detailed_operation_counts[detailed_key] += 1
        else:
            self.detailed_operation_counts[detailed_key] = 1
    def log_detailed_operation_counts(self):
        """打印操作的详细统计"""
        self._print("\nDetailed operation breakdown:")
        for operation, count in self.detailed_operation_counts.items():
            self._print(f"{operation}: {count} times")

    def print_standard_info(self, server_id, datacenter_id, server_generation, operation):
        """标准信息输出，调用一次即可"""
        standard_info = f"[ServerID: {server_id}] [DataCenterID: {datacenter_id}] [server_generation: {server_generation}] [Operation: {operation}]"
        self._print(standard_info)
        logger.debug(standard_info)

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
            logger.debug(" ".join(map(str, args)))

    def choose_operation(self):
        operations = list(self.operation_probabilities.keys())
        probabilities = list(self.operation_probabilities.values())
        return random.choices(operations, weights=probabilities, k=1)[0]

    def update_operation_probabilities(self, operation_type, success):
        if success:
            self.operation_probabilities[operation_type] *= 1.1  
        else:
            self.operation_probabilities[operation_type] *= 0.9  

        total_probability = sum(self.operation_probabilities.values())
        for op in self.operation_probabilities:
            self.operation_probabilities[op] /= total_probability  
            self.operation_probabilities[op] = (
                self.low_pass_filter_alpha * self.operation_probabilities[op] +
                (1 - self.low_pass_filter_alpha) * self.initial_operation_probabilities[op]
            )
        self._print(self.operation_probabilities)

    def generate_neighbor(self, current_solution, servers, datacenters):
        self.operation_type = self.choose_operation()

        if self.operation_type == 'buy':
            self._print("临域生成：进行buy操作")
            solution, errinfo = self.generate_new_purchase(current_solution, servers, datacenters)
        elif self.operation_type == 'adjust_dismiss_time':
            self._print("临域生成：进行dismiss时间调整操作")
            solution, errinfo = self.adjust_dismiss_time(current_solution, servers)
        elif self.operation_type == 'adjust_time_slot':
            self._print("临域生成：进行时间段调整操作")
            solution, errinfo = self.adjust_time_slot(current_solution, servers)
        elif self.operation_type == 'adjust_quantity':
            self._print("临域生成：进行数量调整操作")
            solution, errinfo = self.adjust_quantity(current_solution, servers)
        else:
            solution, errinfo = None, "无效的操作"
            self._print(errinfo)
        
        return solution, errinfo

    def generate_new_purchase(self, current_solution, servers, datacenters):
        time_step = random.randint(1, TIME_STEPS)
        data_center = random.choice(datacenters['datacenter_id'].unique())

        available_servers = servers[(servers['release_time'].apply(lambda x: eval(x)[0]) <= time_step) &
                                    (servers['release_time'].apply(lambda x: eval(x)[1]) >= time_step)]

        if available_servers.empty:
            return None, "无可用服务器"  

        selected_server_type = available_servers.loc[random.choice(available_servers.index)]
        server_generation = selected_server_type['server_generation']
        life_expectancy = selected_server_type['life_expectancy']
        slots_size = selected_server_type['slots_size']

        # 50% 概率决定是否设置随机过期时间
        if random.random() < 0.5:
            # 设置随机的过期时间
            dismiss_life = random.randint(1, life_expectancy)
            self.pending_sub_type = 'custom_life'  # 暂存操作类型
        else:
            # 否则使用默认的最大寿命
            dismiss_life = life_expectancy  # 默认使用最大寿命
            self.pending_sub_type = 'life_expectancy'  # 暂存操作类型

        # 计算基于 dismiss_life 的可用插槽
        max_available_slots = self.slot_manager.get_maximum_available_slots(time_step, time_step + dismiss_life, data_center)

        if max_available_slots >= slots_size:
            max_purchase_count = int(max_available_slots // slots_size * MAX_BUY_PERCENTAGE)  # 限制为可用插槽的一定比例
            purchase_count = random.randint(1, max(1, max_purchase_count))
            total_slots_needed = purchase_count * slots_size
            ID = self.id_gen.next_id()

            # 构造新的解决方案行
            new_rows = [{
                'time_step': time_step,
                'datacenter_id': data_center,
                'server_generation': server_generation,
                'server_id': ID,
                'action': 'buy',
                'quantity': purchase_count,
                'dismiss_life_expectancy': dismiss_life,
            }]
            new_solution = pd.concat([current_solution, pd.DataFrame(new_rows)], ignore_index=True)
            diff_server = ServerInfo(server_id=ID, 
                                     server_generation=server_generation, 
                                     quantity=purchase_count,
                                     dismiss_time=time_step + dismiss_life - 1,
                                     buy_and_move_info=[ServerMoveInfo(time_step=time_step - 1, target_datacenter=data_center)])
            self.S.apply_server_change(diff_server)

            # 记录更新操作，不立即执行
            self.pending_updates = [(time_step, time_step + dismiss_life, data_center, total_slots_needed, 'buy')]
            self.print_standard_info(server_id=ID, datacenter_id=data_center, server_generation=server_generation, operation="buy")
            self._print(f"购买了 {purchase_count} 台服务器, 时间段 {time_step} - {time_step + dismiss_life}, 过期寿命 {dismiss_life}")
            return new_solution, None
        else:
            return None, f"数据中心 {data_center} 剩余插槽数量 {max_available_slots} 无法选购当前种类服务器，当前需要 {slots_size} 插槽"

    def adjust_dismiss_time(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 深拷贝当前的解，防止修改原解
        new_solution = current_solution.copy(deep=True)

        # 随机选择增加或减少 dismiss 时间
        adjustment_direction = random.choice(["increase", "decrease"])

        # 根据调整方向，过滤可以操作的购买操作
        buy_actions = self._filter_valid_buy_actions(new_solution, servers, adjustment_direction)
        if buy_actions.empty:
            return None, f"没有可{'增加' if adjustment_direction == 'increase' else '减少'}寿命的购买操作"

        # 随机选择一个购买操作
        selected_action_index = random.choice(buy_actions.index)
        selected_action = new_solution.loc[selected_action_index]

        # 调整 dismiss 时间
        return self._adjust_life_expectancy(new_solution, selected_action, servers, adjustment_direction)

    def _filter_valid_buy_actions(self, solution, servers, adjustment_direction):
        """根据调整方向，过滤出有效的购买操作"""
        if adjustment_direction == "decrease":
            # 过滤出可以减少 dismiss 寿命的购买操作
            return solution[(solution['action'] == 'buy') & (solution['dismiss_life_expectancy'] > 1)]
        elif adjustment_direction == "increase":
            # 过滤出可以增加 dismiss 寿命的购买操作
            return solution[solution.apply(
                lambda row: row['action'] == 'buy' and row['dismiss_life_expectancy'] < servers.loc[servers['server_generation'] == row['server_generation'], 'life_expectancy'].values[0],
                axis=1
            )]

    def _adjust_life_expectancy(self, new_solution, selected_action, servers, adjustment_direction):
        """根据调整方向，修改 dismiss_life_expectancy 并更新插槽"""
        buy_time_step = selected_action['time_step']
        server_generation = selected_action['server_generation']
        current_dismiss_life = selected_action['dismiss_life_expectancy']
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]
        total_slots_needed = selected_action['quantity'] * slots_size
        real_dismiss_time = min(buy_time_step + current_dismiss_life, TIME_STEPS + 1)

        if adjustment_direction == "decrease":
            # 减少 dismiss 时间
            new_dismiss_life = random.randint(1, current_dismiss_life - 1)
            new_real_dismiss_time = buy_time_step + new_dismiss_life
            new_solution.at[selected_action.name, 'dismiss_life_expectancy'] = new_dismiss_life
            if new_real_dismiss_time < real_dismiss_time:
                self.pending_updates.append((new_real_dismiss_time, real_dismiss_time, selected_action['datacenter_id'], total_slots_needed, 'cancel'))
            self.print_standard_info(selected_action['server_id'], selected_action['datacenter_id'], server_generation, "dismiss")
            self._print(f"减少了购买操作的截止时间: 购买时间 {buy_time_step}, 新的截止时间: {new_real_dismiss_time}, 新的裁撤寿命: {new_dismiss_life}")
            server = copy.deepcopy(self.S.server_map[selected_action['server_id']])
            server.dismiss_time = new_real_dismiss_time - 1
            self.S.apply_server_change(server)


        elif adjustment_direction == "increase":
            # 获取服务器的最大寿命
            max_life_expectancy = servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0]
            new_dismiss_life = random.randint(current_dismiss_life + 1, max_life_expectancy)
            new_real_dismiss_time = buy_time_step + new_dismiss_life
            new_solution.at[selected_action.name, 'dismiss_life_expectancy'] = new_dismiss_life
            if new_real_dismiss_time > real_dismiss_time:
                self.pending_updates.append((real_dismiss_time, new_real_dismiss_time, selected_action['datacenter_id'], total_slots_needed, 'buy'))
            self.print_standard_info(selected_action['server_id'], selected_action['datacenter_id'], server_generation, "dismiss")
            self._print(f"增加了购买操作的截止时间: 购买时间 {buy_time_step}, 新的截止时间: {new_real_dismiss_time}, 新的裁撤寿命: {new_dismiss_life}")
            
            server = copy.deepcopy(self.S.server_map[selected_action['server_id']])
            server.dismiss_time = new_real_dismiss_time - 1
            self.S.apply_server_change(server)
        return new_solution, None

    def adjust_time_slot(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 深拷贝当前解，防止修改原解
        new_solution = current_solution.copy(deep=True)

        # 随机选择一个现有的购买操作
        buy_actions = new_solution[new_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        selected_action_index = random.choice(buy_actions.index)
        selected_action = new_solution.loc[selected_action_index]

        # 获取服务器的当前时间步和自定义寿命
        old_time_step = selected_action['time_step']
        server_generation = selected_action['server_generation']
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]
        
        # 注意此处，使用的是 dismiss_life_expectancy 
        dismiss_life_expectancy = selected_action['dismiss_life_expectancy']

        # 生成新的时间段（随机前移或后移）
        new_time_step, msg = generate_new_time_step(old_time_step, TIME_STEPS)
        if new_time_step is None:
            return None, msg

        if new_time_step < 1 or new_time_step > TIME_STEPS:
            return None, "新时间段超出范围"

        data_center = selected_action['datacenter_id']
        required_slots = slots_size * selected_action['quantity']
        # 使用模拟方法检查新时间段插槽可用性
        if self.slot_manager.simulate_slot_availability(new_time_step, new_time_step + dismiss_life_expectancy, data_center, required_slots, old_time_step, old_time_step + dismiss_life_expectancy):
            # 如果可行，才更新购买操作和插槽使用
            new_solution.at[selected_action_index, 'time_step'] = new_time_step
            server = copy.deepcopy(self.S.server_map[selected_action['server_id']])
            server.buy_and_move_info[0].time_step = new_time_step - 1
            # 同时根据 new_time_step 调整的程度，调整 dismiss_time
            server.dismiss_time = new_time_step + dismiss_life_expectancy - 1
            self.S.apply_server_change(server)

            # 更新插槽占用，记录需要的插槽更新操作
            self.pending_updates.append((old_time_step, old_time_step + dismiss_life_expectancy, data_center, required_slots, 'cancel'))
            self.pending_updates.append((new_time_step, new_time_step + dismiss_life_expectancy, data_center, required_slots, 'buy'))

            self.print_standard_info(server_id=selected_action['server_id'], datacenter_id=data_center, server_generation=server_generation, operation="time_slot")
            self._print(f"调整了购买操作的时间段: 原时间段 {old_time_step}, 新时间段 {new_time_step}")
            return new_solution, None
        else:
            available_slots_new = self.slot_manager.get_maximum_available_slots(new_time_step, new_time_step + dismiss_life_expectancy, data_center)
            available_slots_old = self.slot_manager.get_maximum_available_slots(old_time_step, old_time_step + dismiss_life_expectancy, data_center)
            
            self._print(
                f"新的时间段插槽不足: "
                f"原时间段可用插槽: {available_slots_old}, 新时间段可用插槽: {available_slots_new}, "
                f"所需插槽: {required_slots}, "
                f"数据中心: {data_center}, "
                f"旧时间段: {old_time_step}-{old_time_step + dismiss_life_expectancy}, "
                f"新时间段: {new_time_step}-{new_time_step + dismiss_life_expectancy}"
            )
            return None, "新的时间段插槽不足"


    def adjust_quantity(self, current_solution, servers):
        # 如果当前解是空的，直接返回
        if current_solution.empty:
            return None, "空解"

        # 深拷贝当前的解，防止修改原解
        new_solution = current_solution.copy(deep=True)

        # 随机选择一个现有的购买操作
        buy_actions = new_solution[new_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        # 随机选择一个购买操作
        selected_action_index = random.choice(buy_actions.index)
        selected_action = new_solution.loc[selected_action_index]

        # 根据server_generation获取对应的slots_size
        server_generation = selected_action['server_generation']
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]

        # 获取当前购买数量
        current_quantity = selected_action['quantity']
        data_center = selected_action['datacenter_id']

        # 随机决定增加或减少购买数量
        adjustment_direction = random.choice(["increase", "decrease"])

        if adjustment_direction == "decrease":
            # 如果当前购买数量已经为1，无法再减少
            if current_quantity <= 1:
                self._print(f"无法调整服务器 {selected_action['server_id']} 到最小值：当前购买数量为 {current_quantity}")
                return None, "无法调整到最小值，已经是1"

            # 随机减少一定数量，但不能小于1
            min_quantity = max(1, int (current_quantity * 0.5))  # 减少的最小范围为一半
            new_quantity = random.randint(min_quantity, current_quantity - 1)
            reduction_amount = current_quantity - new_quantity

            # 记录需要释放的插槽数（延迟执行）
            total_slots_to_restore = reduction_amount * slots_size
            time_step = selected_action['time_step']

            # 更新购买数量
            new_solution.at[selected_action_index, 'quantity'] = new_quantity
            dismiss_life_expectancy = selected_action['dismiss_life_expectancy']
            # 记录更新操作，不立即执行
            self.pending_updates.append(
                (time_step, time_step + dismiss_life_expectancy,
                data_center, total_slots_to_restore, 'cancel')
            )
            self.pending_sub_type = 'decrease'  # 暂存类型
            self.print_standard_info(server_id=selected_action['server_id'], datacenter_id=data_center, server_generation=server_generation, operation="quantity")
            self._print(f"减少了 {reduction_amount} 个服务器, 新数量 {new_quantity}")

            server = copy.deepcopy(self.S.server_map[selected_action['server_id']])
            server.quantity = new_quantity
            self.S.apply_server_change(server)

        elif adjustment_direction == "increase":
            # 获取当前时间段内数据中心的可用插槽数
            available_slots = self.slot_manager.get_maximum_available_slots(
                selected_action['time_step'], 
                selected_action['time_step'] + servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0],
                data_center
            )

            # 计算可以增加的最大服务器数量，确保不会超过可用插槽
            max_additional_servers = int(available_slots // slots_size * MAX_ADD_PERCENTAGE)  # 最大增加数量为可用插槽的一定比例
            if max_additional_servers <= 0:
                self._print(
                    f"无法增加服务器数量: "
                    f"当前可用插槽: {available_slots}, "
                    f"每台服务器所需插槽: {slots_size}, "
                    f"当前服务器数量: {current_quantity}, "
                    f"数据中心: {data_center}, "
                    f"时间步长: {selected_action['time_step']}"
                )
                return None, "无法增加，插槽不足"

            # 随机增加一定数量，但最多增加到可用插槽允许的最大数量
            new_quantity = random.randint(current_quantity + 1, current_quantity + max_additional_servers)

            increase_amount = new_quantity - current_quantity
            total_slots_needed = increase_amount * slots_size

            # 更新购买数量
            new_solution.at[selected_action_index, 'quantity'] = new_quantity
            dismiss_life_expectancy = selected_action['dismiss_life_expectancy']
            # 更新插槽使用情况，记录更新操作，不立即执行
            time_step = selected_action['time_step']
            self.pending_updates.append(
                (time_step, time_step + dismiss_life_expectancy,
                data_center, total_slots_needed, 'buy')
            )
            self.pending_sub_type = 'increase'  # 暂存类型
            self.print_standard_info(server_id=selected_action['server_id'], datacenter_id=data_center, server_generation=server_generation, operation="quantity")
            self._print(f"增加了 {increase_amount} 个服务器, 新数量 {new_quantity}")
            server = copy.deepcopy(self.S.server_map[selected_action['server_id']])
            server.quantity = new_quantity
            self.S.apply_server_change(server)

        return new_solution, None

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def clean_loop_cache(self):
        self.pending_updates.clear()
        self.pending_sub_type = ""

    def run(self, S, servers, datacenters):
        current_score = self.evaluate(S)
        best_score = current_score

        for i in range(self.max_iter):
            self._print(f"<------------- Iteration {i}/温度最大步数{self.step_count}/固定最大步数{self.max_iter} ------------->")
            self.clean_loop_cache()
            self.current_step = i
            # check_result = self.slot_manager.check_slot_consistency(self.current_solution, servers)
            new_solution, errinfo = self.generate_neighbor(self.current_solution, servers, datacenters)
            if new_solution is not None:
                # new_input = DiffInput(is_new=True, step=1, diff_solution=new_solution)
                #new_score = S.SA_evaluation_function(new_input)
                new_score = self.S.diff_evaluation()
                # print(f"new_score: {new_score}, new_score2: {new_score2}")
                accept_prob = self.acceptance_probability(current_score, new_score)

                if accept_prob != 1.0:
                    self._print(f'Acceptance Probability: {accept_prob:.2f}')

                if accept_prob == 1.0 or random.random() < accept_prob:
                    self.current_solution = new_solution
                    current_score = new_score
                    self._print(f'\033[93mAccepted\033[0m new solution with score: {current_score:.5e}')

                    self.update_slot()
                    self.S.commit_server_changes()
                    check_result = self.slot_manager.check_slot_consistency(self.current_solution, servers)
                    self._print(f"插槽一致性检查结果: {check_result}")
                    if check_result is False:
                        raise ValueError("插槽一致性检查失败")

                    if new_score > best_score:
                        self.update_operation_probabilities(self.operation_type, True)
                        self.best_solution = copy.deepcopy(new_solution)
                        best_score = new_score
                        print(f'\033[92mNew best solution for {self.seed} found with score: {best_score:.5e}\033[0m')

                        # self.operation_success_counts[self.operation_type] += 1
                        self.log_detailed_operation(self.operation_type, self.pending_sub_type)

                else:
                    self.update_operation_probabilities(self.operation_type, False)
                    self._print(f'\033[91mRejected\033[0m new solution with score: {new_score:.5e}')
                    self.S.discard_server_changes()

                self.current_temp *= self.alpha
                if self.current_temp < self.min_temp:
                    break
            else:
                self.update_operation_probabilities(self.operation_type, False)
                self._print(f'\033[94mNo valid neighbor found. {errinfo}\033[0m')

        self.log_detailed_operation_counts()  # 输出详细的操作统计
        self._print(f'Best solution found with score: {best_score:.5e}')
        return self.best_solution, best_score

    def update_slot(self):
        for update in self.pending_updates:
            self.slot_manager.update_slots(*update[:-1], operation=update[-1])
        self.pending_updates.clear()

    def evaluate(self, S: DiffSolution):
        input = DiffInput(is_new=True, step=1, diff_solution=self.current_solution)
        return S.SA_evaluation_function(input)


def get_my_solution(seed, verbose=False):
    # 加载服务器和数据中心数据
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')
    # 初始化插槽管理器
    slot_manager = SlotAvailabilityManager(datacenters, verbose=verbose)
    # 初始化解决方案数据框
    initial_solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'quantity'])
    initial_solution['dismiss_life_expectancy'] = pd.Series(dtype='int')
    # 初始化退火算法和差分解
    S = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(initial_solution, slot_manager=slot_manager, seed=seed, initial_temp=100000, min_temp=100, alpha=0.999, max_iter=2500, verbose=verbose)
    # 运行退火算法，获取最佳解决方案
    best_solution, best_score = sa.run(S, servers, datacenters)

    expanded_rows = []
    dismiss_operations = []

    for _, row in best_solution.iterrows():
        for i in range(row['quantity']):
            new_row = row.drop('quantity').copy()
            new_row['server_id'] = f"{row['server_id']}:{i + 1}"
            expanded_rows.append(new_row)

            # 计算真实的解散时间（基于购买时间步长和 dismiss 寿命）
            real_dismiss_time = row['time_step'] + row['dismiss_life_expectancy']

            # 如果dismiss时间小于等于TIME_STEPS，创建dismiss操作
            if row['dismiss_life_expectancy'] <= TIME_STEPS:
                dismiss_row = new_row.copy()
                dismiss_row['action'] = 'dismiss'
                dismiss_row['time_step'] = real_dismiss_time
                dismiss_operations.append(dismiss_row)

    expanded_df = pd.DataFrame(expanded_rows)
    dismiss_df = pd.DataFrame(dismiss_operations)

    # 将buy和dismiss操作合并
    final_df = pd.concat([expanded_df, dismiss_df], ignore_index=True)
    final_df = final_df.drop(columns=['dismiss_life_expectancy'], errors='ignore')

    save_solution(final_df, f'./output/{seed}_{best_score:.5e}.json')
    save_solution(best_solution, f'./output/quantity_{seed}_{best_score:.5e}.json')
    print(f'Final best solution:\n{best_solution}')

    return

if __name__ == '__main__':
    start = time.time()
    seed = 3329
    get_my_solution(seed, verbose=True)
    end = time.time()
    print(f"耗时: {end - start:.4f} 秒")
