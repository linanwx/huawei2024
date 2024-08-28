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

    def generate_neighbor(self, current_solution, slot_manager:SlotAvailabilityManager, servers, datacenters):
        # 随机选择一个时间步和数据中心
        time_step = random.randint(1, TIME_STEPS)
        data_center = random.choice(datacenters['datacenter_id'].unique())

        # 在指定的时间步内，选择一个符合条件的服务器生成新的购买操作
        available_servers = servers[(servers['release_time'].apply(lambda x: eval(x)[0]) <= time_step) &
                                    (servers['release_time'].apply(lambda x: eval(x)[1]) >= time_step)]
        
        if available_servers.empty:
            return None  # 无可用服务器
        
        # 随机选择一种服务器类型
        selected_server_type = available_servers.sample().iloc[0]
        server_generation = selected_server_type['server_generation']
        life_expectancy = selected_server_type['life_expectancy']
        slots_size = selected_server_type['slots_size']

        # 计算这个时间段内数据中心的最大可用插槽数
        max_available_slots = slot_manager.get_maximum_available_slots(time_step, time_step + life_expectancy, data_center)
        

        # 计算批量购买的最大数量
        if max_available_slots >= slots_size:
            max_purchase_count = max_available_slots // slots_size // 4 # 最多购买一半的插槽
            if max_purchase_count < 1:
                return None
            purchase_count = random.randint(1, max_purchase_count)
            total_slots_needed = purchase_count * slots_size
            new_rows = []
            # for _ in range(purchase_count):
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
            self.pending_updates = [(time_step, time_step + life_expectancy, data_center, total_slots_needed)]
            return new_solution
        else:
            return None

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def run(self, S, servers, datacenters):
        current_score = self.evaluate(S)
        best_score = current_score

        for i in range(self.max_iter):
            new_solution = self.generate_neighbor(self.current_solution, slot_manager, servers, datacenters)
            if new_solution is not None:
                new_input = DiffInput(is_new=True, step=1, diff_solution=new_solution)
                new_score = S.SA_evaluation_function(new_input)
                accept_prob = self.acceptance_probability(current_score, new_score)
                print(f'Iteration {i}, Acceptance Probability: {accept_prob:.4f}')

                if random.random() < accept_prob:
                    self.current_solution = new_solution
                    current_score = new_score
                    print(f'Iteration {i}, \033[93mAccepted\033[0m new solution with score: {current_score}')

                    # 执行延迟的更新操作
                    for update in self.pending_updates:
                        slot_manager.update_slots(*update, operation='buy')
                    self.pending_updates.clear()  # 清空更新队列

                    if new_score > best_score:
                        self.best_solution = copy.deepcopy(new_solution)
                        best_score = new_score
                        print(f'\033[92mNew best solution found with score: {best_score}\033[0m')
                else:
                    print(f'Iteration {i}, \033[91mRejected\033[0m new solution with score: {new_score}')

                self.current_temp *= self.alpha
                if self.current_temp < self.min_temp:
                    break
            else:
                print(f'Iteration {i}, \033[94mNo valid neighbor found.\033[0m')

        print(f'Best solution found with score: {best_score}')
        return self.best_solution, best_score

    def evaluate(self, S:DiffSolution):
        input = DiffInput(is_new=True, step=1, diff_solution=self.current_solution)
        return S.SA_evaluation_function(input)


# 主程序
if __name__ == '__main__':
    seed = 123
    # 加载数据
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')

    # 初始化插槽可用性管理器
    slot_manager = SlotAvailabilityManager(datacenters)

    # 初始化模拟退火算法
    initial_solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'quantity'])
    S = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(initial_solution, initial_temp=10000, min_temp=1, alpha=0.99, max_iter=50)

    # 运行算法
    best_solution, best_score = sa.run(S, servers, datacenters)

    # 展开DataFrame
    expanded_rows = []
    for _, row in best_solution.iterrows():
        for i in range(row['quantity']):
            # 创建新行，移除 quantity 属性并调整 server_id
            new_row = row.drop('quantity').copy()
            new_row['server_id'] = f"{row['server_id']}_{i + 1}"
            expanded_rows.append(new_row)

    # 构建最终的DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    save_solution(expanded_df, f'./output/{seed}_{best_score}.json')
    save_solution(best_solution, f'./output/quantity_{seed}_{best_score}.json')
    print(f'Final best solution: {best_solution}')
