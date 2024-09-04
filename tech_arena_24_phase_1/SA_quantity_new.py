import math
import numpy as np
import pandas as pd
import random
import copy

from evaluation_diff_quantity import DiffInput, DiffSolution
from utils import save_solution

TIME_STEPS = 168 

import threading

class ThreadSafeIDGenerator:
    def __init__(self, start=0):
        self.current_id = start
        self.lock = threading.Lock()

    def next_id(self):
        with self.lock:
            self.current_id += 1
            return str(self.current_id)

# 插槽可用性管理器
class SlotAvailabilityManager:
    def __init__(self, datacenters):
        # 初始化插槽可用性表
        self.slots_table = pd.DataFrame({
            'time_step': np.arange(1, TIME_STEPS + 1)
        })
        for dc in datacenters['datacenter_id']:
            self.slots_table[dc] = datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0]

    def check_availability(self, start_time, end_time, data_center, slots_needed):
        end_time = min(end_time, TIME_STEPS)
        for ts in range(start_time, end_time + 1):
            if self.slots_table.at[ts - 1, data_center] < slots_needed:
                return False
        return True

    def get_maximum_available_slots(self, start_time, end_time, data_center):
        end_time = min(end_time, TIME_STEPS)
        # 返回给定时间段内，指定数据中心的最小可用插槽数
        available_slots = self.slots_table.loc[start_time-1:end_time-1, data_center].min()
        return available_slots

    def update_slots(self, start_time, end_time, data_center, slots_needed, operation='buy'):
        end_time = min(end_time, TIME_STEPS)
        for ts in range(start_time, end_time + 1):
            if operation == 'buy':
                self.slots_table.at[ts - 1, data_center] -= slots_needed
            elif operation == 'cancel':
                self.slots_table.at[ts - 1, data_center] += slots_needed

    def simulate_slot_availability(self, start_time, end_time, data_center, slots_needed, existing_start, existing_end):
        """
        模拟调整时间段后的插槽可用性检查，不实际更改插槽表。
        """
        end_time = min(end_time, TIME_STEPS)
        existing_end = min(existing_end, TIME_STEPS)

        for ts in range(start_time, end_time + 1):
            simulated_slots = self.slots_table.at[ts - 1, data_center]
            
            # 如果时间段内是已经被占用的部分，则先加回原来占用的插槽
            if existing_start <= ts <= existing_end:
                simulated_slots += slots_needed

            # 检查新的时间段是否可行
            if simulated_slots < slots_needed:
                return False

        return True
    
def optimized_linear_biased_time_step(start=1, end=168):
    # Generate a single uniformly distributed random number
    uniform_random = np.random.rand()
    
    # Apply the linear bias (strength 2)
    linear_decay = (1 - uniform_random) ** 2
    
    # Map to the time steps range
    time_step = int(start + linear_decay * (end - start))
    
    return time_step

# 模拟退火算法实现
class SimulatedAnnealing:
    def __init__(self, initial_solution, initial_temp, min_temp, alpha, max_iter):
        self.current_solution = initial_solution
        self.best_solution = copy.deepcopy(initial_solution)
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.id_gen = ThreadSafeIDGenerator(start=0)
        self.pending_updates = []  # 用于延迟更新的操作队列

        # 初始化操作概率表
        self.operation_probabilities = {
            'buy': 1/3,
            # 'adjust_time_slot': 1/3,
            # 'replace_server': 1/3
        }
        self.low_pass_filter_alpha = 0.9  # 低通滤波参数

        self.step_count = int(math.log(min_temp / initial_temp) / math.log(alpha))
        print(f"将在 {self.step_count} 步中完成退火过程")

    def choose_operation(self):
        operations = list(self.operation_probabilities.keys())
        probabilities = list(self.operation_probabilities.values())
        return random.choices(operations, weights=probabilities, k=1)[0]
    
    def update_operation_probabilities(self, operation_type, success):
        # 基于操作成功与否调整概率
        if success:
            self.operation_probabilities[operation_type] *= 1.1  # 增加成功操作的概率
        else:
            self.operation_probabilities[operation_type] *= 0.9  # 减少失败操作的概率

        # 确保概率之和为1，并应用低通滤波
        total_probability = sum(self.operation_probabilities.values())
        for op in self.operation_probabilities:
            self.operation_probabilities[op] /= total_probability  # 归一化
            self.operation_probabilities[op] = (
                self.low_pass_filter_alpha * self.operation_probabilities[op] +
                (1 - self.low_pass_filter_alpha) * (1 / len(self.operation_probabilities))
            )
        
        print(self.operation_probabilities)

    def generate_neighbor(self, current_solution, slot_manager: SlotAvailabilityManager, servers, datacenters):
        # 随机选择操作类型：购买新的服务器或调整现有购买数量
        self.operation_type = self.choose_operation()
        # operation_type = random.choice(['buy'])

        if self.operation_type == 'buy':
            print("临域生成：进行buy操作")
            solution, errinfo = self.generate_new_purchase(current_solution, servers, datacenters)
            if errinfo != None:
                print(errinfo)
        elif self.operation_type == 'adjust':
            print("临域生成：进行adjust操作")
            solution, errinfo = self.decrease_existing_purchase(current_solution, servers)
            if errinfo != None:
                print(errinfo)
        elif self.operation_type == 'adjust_time_slot':
            print("临域生成：进行时间段调整操作")
            solution, errinfo = self.adjust_time_slot(current_solution, slot_manager, servers, datacenters)
            if errinfo != None:
                print(errinfo)
        elif self.operation_type == 'replace_server':
            print("临域生成：进行服务器替换操作")
            solution, errinfo = self.replace_server(current_solution, servers)
            if errinfo != None:
                print(errinfo)
        
        return solution
    def replace_server(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 随机选择一个现有的购买操作
        buy_actions = current_solution[current_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        selected_action_index = buy_actions.sample().index[0]
        selected_action = current_solution.loc[selected_action_index]

        # 获取当前服务器的基本信息
        old_server_generation = selected_action['server_generation']
        slots_size = servers.loc[servers['server_generation'] == old_server_generation, 'slots_size'].values[0]
        time_step = selected_action['time_step']

        # 查找所有在同一时间段内可用的且插槽大小相同的服务器，但排除当前的服务器类型
        available_servers = servers[(servers['slots_size'] == slots_size) & 
                                    (servers['server_generation'] != old_server_generation) &  # 排除相同的服务器类型
                                    (servers['release_time'].apply(lambda x: eval(x)[0]) <= time_step) &
                                    (servers['release_time'].apply(lambda x: eval(x)[1]) >= time_step)]

        if available_servers.empty:
            return None, "无可用的替换服务器"  # 无可用的替换服务器
        
        # 随机选择一种新的服务器类型作为替换
        new_server_type = available_servers.sample().iloc[0]
        new_server_generation = new_server_type['server_generation']

        # 替换操作
        current_solution.at[selected_action_index, 'server_generation'] = new_server_generation

        print(f"服务器 {selected_action['server_id']} 被替换为 {new_server_generation}")

        return current_solution, None
    
    def biased_time_step(self, start=1, end=TIME_STEPS, scale=0.1):
        """
        根据指数分布生成一个偏向早期的时间步长，scale 控制偏斜程度。
        start: 时间步的起始点。
        end: 时间步的结束点。
        scale: 控制指数分布的偏斜程度，值越小，越偏向早期时间。
        """
        # 生成一个符合指数分布的随机数
        exp_random = np.random.exponential(scale)
        
        # 将指数分布结果映射到指定的时间步区间内
        time_step = int(start + (end - start) * exp_random)
        
        # 确保时间步在合法范围内
        time_step = max(start, min(end, time_step))
        
        return time_step

    def generate_new_purchase(self, current_solution, servers, datacenters):
        # 随机选择一个时间步和数据中心
        time_step = random.randint(1, TIME_STEPS)
        print(f"操作点位 {time_step}")
        data_center = random.choice(datacenters['datacenter_id'].unique())
        print(f"操作服务器 {data_center}")

        # 在指定的时间步内，选择一个符合条件的服务器生成新的购买操作
        available_servers = servers[(servers['release_time'].apply(lambda x: eval(x)[0]) <= time_step) &
                                    (servers['release_time'].apply(lambda x: eval(x)[1]) >= time_step)]
        
        if available_servers.empty:
            return None, "无可用服务器"  # 无可用服务器
        
        # 随机选择一种服务器类型
        selected_server_type = available_servers.sample().iloc[0]
        server_generation = selected_server_type['server_generation']
        life_expectancy = selected_server_type['life_expectancy']
        slots_size = selected_server_type['slots_size']

        # 计算这个时间段内数据中心的最大可用插槽数
        max_available_slots = slot_manager.get_maximum_available_slots(time_step, time_step + life_expectancy, data_center)
        
        # 计算批量购买的最大数量
        if max_available_slots >= slots_size:
            max_purchase_count = max_available_slots // slots_size // 8 # 最多购买一定比例的插槽， 不要全部占了
            if max_purchase_count < 1:
                max_purchase_count = 1
            purchase_count = random.randint(1, max_purchase_count)
            total_slots_needed = purchase_count * slots_size
            new_rows = []
            new_row = {
                'time_step': time_step,
                'datacenter_id': data_center,
                'server_generation': server_generation,
                'server_id': self.id_gen.next_id(),  # 生成唯一ID
                'action': 'buy',
                'quantity': purchase_count
            }
            new_rows.append(new_row)
            new_solution = pd.concat([current_solution, pd.DataFrame(new_rows)], ignore_index=True)
            # 延迟更新，先记录更新操作
            self.pending_updates = [(time_step, time_step + life_expectancy, data_center, total_slots_needed, 'buy')]
            return new_solution, None
        else:
            return None, f"剩余插槽数量无法选购当前种类服务器，当前需要{slots_size}"
        
    def adjust_time_slot(self, current_solution, slot_manager: SlotAvailabilityManager, servers, datacenters):
        if current_solution.empty:
            return None, "空解"

        # 随机选择一个现有的购买操作
        buy_actions = current_solution[current_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        selected_action_index = buy_actions.sample().index[0]
        selected_action = current_solution.loc[selected_action_index]

        # 获取服务器的当前时间步和生存期
        old_time_step = selected_action['time_step']
        server_generation = selected_action['server_generation']
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]
        life_expectancy = servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0]

        # 生成新的时间段（随机前移或后移）
        new_time_step = old_time_step + random.choice([-10, 10])

        # 确保新的时间段在有效范围内
        # if new_time_step < 1 or new_time_step + life_expectancy > TIME_STEPS:
        if new_time_step < 1 or new_time_step > TIME_STEPS:
            return None, "新时间段超出范围"

        data_center = selected_action['datacenter_id']
        
        # 使用模拟方法检查新时间段插槽可用性
        if slot_manager.simulate_slot_availability(new_time_step, new_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], old_time_step, old_time_step + life_expectancy):
            # 如果可行，才更新购买操作和插槽使用
            current_solution.at[selected_action_index, 'time_step'] = new_time_step

            # 更新插槽占用，记录需要的插槽更新操作
            self.pending_updates.append((old_time_step, old_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], 'cancel'))
            self.pending_updates.append((new_time_step, new_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], 'buy'))

            return current_solution, None
        else:
            return None, "新的时间段插槽不足"
    
    def decrease_existing_purchase(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 随机选择一个现有的购买操作
        buy_actions = current_solution[current_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        selected_action_index = buy_actions.sample().index[0]
        selected_action = current_solution.loc[selected_action_index]

        # 根据server_generation获取对应的slots_size
        server_generation = selected_action['server_generation']
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]

        # 计算可调整的最大数量
        current_quantity = selected_action['quantity']

        # 随机调小一定比例
        min_quantity = current_quantity // 2
        new_quantity = random.randint(min_quantity, current_quantity - 1)

        if current_quantity <= 1:
            return None, "无法调整到最小值，已经是1"  # 如果数量已经是1，无法再调小
        
        reduction_amount = current_quantity - new_quantity

        # 记录需要恢复的插槽数（延迟执行）
        time_step = selected_action['time_step']
        data_center = selected_action['datacenter_id']
        total_slots_to_restore = reduction_amount * slots_size

        # 更新购买数量
        current_solution.at[selected_action_index, 'quantity'] = new_quantity

        # 记录更新操作，不立即执行
        self.pending_updates.append((time_step, time_step + servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0], data_center, total_slots_to_restore, 'cancel'))
        print({f"cancel {selected_action['server_id']} count {reduction_amount}"})

        return current_solution, None

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def run(self, S, servers, datacenters):
        current_score = self.evaluate(S)
        best_score = current_score

        for i in range(self.max_iter):
            self.current_step = i
            new_solution = self.generate_neighbor(self.current_solution, slot_manager, servers, datacenters)
            if new_solution is not None:
                new_input = DiffInput(is_new=True, step=1, diff_solution=new_solution)
                new_score = S.SA_evaluation_function(new_input)
                accept_prob = self.acceptance_probability(current_score, new_score)
                if accept_prob != 1.0:
                    print(f'Iteration {i}/{self.step_count}, Acceptance Probability: {accept_prob:.2f}')

                if accept_prob == 1.0 or random.random() < accept_prob:
                    self.current_solution = new_solution
                    current_score = new_score
                    print(f'Iteration {i}, \033[93mAccepted\033[0m new solution with score: {current_score:.5e}')

                    # 执行延迟的更新操作
                    for update in self.pending_updates:
                        slot_manager.update_slots(*update[:-1], operation=update[-1])
                    self.pending_updates.clear()  # 清空更新队列

                    if new_score > best_score:
                        self.update_operation_probabilities(self.operation_type, True)
                        self.best_solution = copy.deepcopy(new_solution)
                        best_score = new_score
                        print(f'\033[92mNew best solution found with score: {best_score:.5e}\033[0m')
                else:
                    self.update_operation_probabilities(self.operation_type, False)
                    print(f'Iteration {i}, \033[91mRejected\033[0m new solution with score: {new_score:.5e}')

                self.current_temp *= self.alpha
                if self.current_temp < self.min_temp:
                    break
            else:
                self.update_operation_probabilities(self.operation_type, False)
                print(f'Iteration {i}, \033[94mNo valid neighbor found.\033[0m')

        print(f'Best solution found with score: {best_score:.5e}')
        return self.best_solution, best_score

    def evaluate(self, S:DiffSolution):
        input = DiffInput(is_new=True, step=1, diff_solution=self.current_solution)
        return S.SA_evaluation_function(input)


# 主程序
if __name__ == '__main__':
    seed = 6053
    # 加载数据
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')

    # 初始化插槽可用性管理器
    slot_manager = SlotAvailabilityManager(datacenters)

    # 初始化模拟退火算法
    initial_solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'quantity'])
    S = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(initial_solution, initial_temp=1000000, min_temp=100, alpha=0.998, max_iter=100)

    # 运行算法
    best_solution, best_score = sa.run(S, servers, datacenters)

    # 展开DataFrame
    expanded_rows = []
    for _, row in best_solution.iterrows():
        for i in range(row['quantity']):
            # 创建新行，移除 quantity 属性并调整 server_id
            new_row = row.drop('quantity').copy()
            new_row['server_id'] = f"{row['server_id']}:{i + 1}"
            expanded_rows.append(new_row)

    # 构建最终的DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    save_solution(expanded_df, f'./output/{seed}_{best_score:.5e}.json')
    save_solution(best_solution, f'./output/quantity_{seed}_{best_score:.5e}.json')
    print(f'Final best solution: {best_solution}')
