import math
import numpy as np
import pandas as pd
import random
import copy

from evaluation_diff_quantity import DiffInput, DiffSolution
from utils import save_solution

TIME_STEPS = 168 
DISMISS_MAX = 10000

import threading

def generate_new_time_step(old_time_step, TIME_STEPS):
    # 计算出有效的可移动范围
    min_shift = max(-50, 1 - old_time_step)  # 确保不小于1
    max_shift = min(50, TIME_STEPS - old_time_step)  # 确保不大于TIME_STEPS

    # 如果移动范围无效，返回None
    if min_shift > max_shift:
        return None, "无法生成有效的新时间段"

    # 在有效范围内随机生成新的时间段
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

# 插槽可用性管理器
class SlotAvailabilityManager:
    def __init__(self, datacenters):
        # 初始化插槽可用性表
        self.slots_table = pd.DataFrame({
            'time_step': np.arange(1, TIME_STEPS + 1)
        })
        for dc in datacenters['datacenter_id']:
            self.slots_table[dc] = datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0]
        self.total_slots = {dc: datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0] for dc in datacenters['datacenter_id']}

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
        # 监控插槽余量百分比日志
        available_slots = self.slots_table[data_center].min()
        total_slots = self.total_slots[data_center]
        used_percentage = 100 * (total_slots - available_slots) / total_slots
        # print(f"Data Center {data_center} Slot Usage: {used_percentage:.2f}%")

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
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    def __init__(self, initial_solution, slot_manager, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        self.current_solution = initial_solution
        self.best_solution = copy.deepcopy(initial_solution)
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.id_gen = ThreadSafeIDGenerator(start=0)
        self.pending_updates = []  # 用于延迟更新的操作队列
        self.slot_manager = slot_manager
        self.verbose = verbose
        self.seed = seed

        # 初始化操作概率表
        self.operation_probabilities = {
            'buy': 25,
            'adjust_dismiss_time': 10,
            'adjust_time_slot': 0,
            'replace_server': 0,
            "adjust": 0
        }
        self.low_pass_filter_alpha = 0.99  # 低通滤波参数
        self.initial_operation_probabilities = copy.deepcopy(self.operation_probabilities)

        self.step_count = int(math.log(min_temp / initial_temp) / math.log(alpha))
        self.print(f"将在 {self.step_count} 步中完成退火过程")

        # 添加操作成功导致分数上升的次数统计
        self.operation_success_counts = {op: 0 for op in self.operation_probabilities.keys()}

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

        # 确保概率之和为1，并应用低通滤波，同时逐步回归初始值
        total_probability = sum(self.operation_probabilities.values())
        for op in self.operation_probabilities:
            self.operation_probabilities[op] /= total_probability  # 归一化
            # 引入回归因子，逐步回归到初始值
            self.operation_probabilities[op] = (
                self.low_pass_filter_alpha * self.operation_probabilities[op] +
                (1 - self.low_pass_filter_alpha) * self.initial_operation_probabilities[op]
            )
        if self.verbose:
            print(self.operation_probabilities)

    def generate_neighbor(self, current_solution, servers, datacenters):
        # 随机选择操作类型：购买新的服务器或调整现有购买数量
        self.operation_type = self.choose_operation()
        self.pending_updates = []  # 清空更新队列

        if self.operation_type == 'buy':
            self.print("临域生成：进行buy操作")
            solution, errinfo = self.generate_new_purchase(current_solution, servers, datacenters)
            if errinfo != None:
                self.print(errinfo)
        elif self.operation_type == 'adjust_dismiss_time':
            self.print("临域生成：进行dismiss时间调整操作")
            solution, errinfo = self.adjust_dismiss_time(current_solution, servers)
            if errinfo != None:
                self.print(errinfo)
        elif self.operation_type == 'adjust':
            self.print("临域生成：进行adjust操作")
            solution, errinfo = self.adjust(current_solution, servers)
            if errinfo != None:
                self.print(errinfo)
        elif self.operation_type == 'adjust_time_slot':
            self.print("临域生成：进行时间段调整操作")
            solution, errinfo = self.adjust_time_slot(current_solution, servers)
            if errinfo != None:
                self.print(errinfo)
        elif self.operation_type == 'replace_server':
            self.print("临域生成：进行服务器替换操作")
            solution, errinfo = self.replace_server(current_solution, servers)
            if errinfo != None:
                self.print(errinfo)
        
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

        if self.verbose:
            print(f"服务器 {selected_action['server_id']} 被替换为 {new_server_generation}")

        return current_solution, None

    def generate_new_purchase(self, current_solution, servers, datacenters):
        # 随机选择一个时间步和数据中心
        time_step = random.randint(1, TIME_STEPS)
        if self.verbose:
            print(f"操作点位 {time_step}")
        data_center = random.choice(datacenters['datacenter_id'].unique())
        if self.verbose:
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
        max_available_slots = self.slot_manager.get_maximum_available_slots(time_step, time_step + life_expectancy, data_center)
        
        # 计算批量购买的最大数量
        if max_available_slots >= slots_size:
            max_purchase_count = max_available_slots // slots_size // 4 # 最多购买一定比例的插槽， 不要全部占了
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
                'quantity': purchase_count,
                'dismiss': DISMISS_MAX,
            }
            new_rows.append(new_row)
            new_solution = pd.concat([current_solution, pd.DataFrame(new_rows)], ignore_index=True)
            # 延迟更新，先记录更新操作
            self.pending_updates = [(time_step, time_step + life_expectancy, data_center, total_slots_needed, 'buy')]
            return new_solution, None
        else:
            return None, f"剩余插槽数量无法选购当前种类服务器，当前需要{slots_size}"
        
    def adjust_time_slot(self, current_solution, servers):
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
        new_time_step = old_time_step + random.choice([-50, 50])

        # 确保新的时间段在有效范围内
        # if new_time_step < 1 or new_time_step + life_expectancy > TIME_STEPS:
        new_time_step, msg = generate_new_time_step(old_time_step, TIME_STEPS)
        if new_time_step == None:
            return None, msg
        if new_time_step < 1 or new_time_step > TIME_STEPS:
            return None, "新时间段超出范围"

        data_center = selected_action['datacenter_id']
        
        # 使用模拟方法检查新时间段插槽可用性
        if self.slot_manager.simulate_slot_availability(new_time_step, new_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], old_time_step, old_time_step + life_expectancy):
            # 如果可行，才更新购买操作和插槽使用
            current_solution.at[selected_action_index, 'time_step'] = new_time_step

            # 更新插槽占用，记录需要的插槽更新操作
            self.pending_updates.append((old_time_step, old_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], 'cancel'))
            self.pending_updates.append((new_time_step, new_time_step + life_expectancy, data_center, slots_size * selected_action['quantity'], 'buy'))

            return current_solution, None
        else:
            return None, "新的时间段插槽不足"
    
    def adjust(self, current_solution, servers):
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

        # 获取当前购买数量
        current_quantity = selected_action['quantity']
        data_center = selected_action['datacenter_id']
        adjustment_direction = random.choice(["increase"])

        if adjustment_direction == "decrease":
            # 随机调小一定比例
            min_quantity = current_quantity // 4
            if min_quantity <= current_quantity - 1:
                return None, "无法调整到最小值，已经是最小值"  # 如果数量已经是最小值，无法再调小
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
            if self.verbose:
                print({f"cancel {selected_action['server_id']} count {reduction_amount}"})
        elif adjustment_direction == "increase":
            # 获取当前数据中心的可用插槽数
            available_slots = self.slot_manager.get_maximum_available_slots(
                selected_action['time_step'], 
                selected_action['time_step'] + servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0],
                data_center
            )

            # 随机增加一定比例，假设最多能增加到当前数量的两倍
            max_quantity = current_quantity * 2  # 这里假设最多增加一倍，实际可以根据需求调整
            new_quantity = random.randint(current_quantity + 1, max_quantity)

            increase_amount = new_quantity - current_quantity
            total_slots_needed = increase_amount * slots_size

            # 检查是否有足够的可用插槽
            if total_slots_needed > available_slots:
                return None, "无法增加，插槽不足"

            # 更新购买数量
            current_solution.at[selected_action_index, 'quantity'] = new_quantity

            # 更新插槽使用情况
            time_step = selected_action['time_step']
            self.pending_updates.append(
                (time_step, time_step + servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0], 
                data_center, total_slots_needed, 'buy')
            )
            if self.verbose:
                print(f"increase {selected_action['server_id']} count {increase_amount}")

        return current_solution, None
    
    def adjust_dismiss_time(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 随机选择一个购买操作
        buy_actions = current_solution[current_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        # 随机选择一个购买操作
        selected_action_index = buy_actions.sample().index[0]
        selected_action = current_solution.loc[selected_action_index]

        # 获取购买时的时间步和服务器的生存期
        buy_time_step = selected_action['time_step']
        server_generation = selected_action['server_generation']
        life_expectancy = servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0]

        # 获取当前的 dismiss 时间，如果它未被修改过，使用购买时间步 + life_expectancy
        current_dismiss_time = selected_action['dismiss'] if selected_action['dismiss'] != DISMISS_MAX else buy_time_step + life_expectancy

        # 确保截止时间不超过最大时间步
        if current_dismiss_time > TIME_STEPS + 1:
            current_dismiss_time = TIME_STEPS + 1

        # 计算最小截止时间
        min_dismiss_time = current_dismiss_time - TIME_STEPS // 5
        if min_dismiss_time < 1:
            min_dismiss_time = 1

        # 在[min_dismiss_time, current_dismiss_time]范围内随机生成新的截止时间
        new_dismiss_time = random.randint(min_dismiss_time, current_dismiss_time)

        # 将新截止时间设置到 selected_action_index['dismiss'] 中
        current_solution.at[selected_action_index, 'dismiss'] = new_dismiss_time

        # 获取插槽使用的服务器相关信息
        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]
        total_slots_needed = selected_action['quantity'] * slots_size

        # 更新插槽，释放在 old dismiss time 到新 dismiss time 间多出来的插槽
        if new_dismiss_time < current_dismiss_time:
            self.pending_updates.append((new_dismiss_time + 1, current_dismiss_time, 
                                        selected_action['datacenter_id'], total_slots_needed, 'cancel'))

        self.print(f"调整了购买操作的截止时间: {selected_action['server_id']}, 原截止时间{current_dismiss_time}新的截止时间: {new_dismiss_time}")

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
            new_solution = self.generate_neighbor(self.current_solution, servers, datacenters)
            if new_solution is not None:
                new_input = DiffInput(is_new=True, step=1, diff_solution=new_solution)
                new_score = S.SA_evaluation_function(new_input)
                accept_prob = self.acceptance_probability(current_score, new_score)
                if accept_prob != 1.0:
                    self.print(f'Iteration {i}/{self.step_count}, Acceptance Probability: {accept_prob:.2f}')

                if accept_prob == 1.0 or random.random() < accept_prob:
                    self.current_solution = new_solution
                    current_score = new_score
                    self.print(f'Iteration {i}, \033[93mAccepted\033[0m new solution with score: {current_score:.5e}')

                    # 执行延迟的更新操作
                    for update in self.pending_updates:
                        self.slot_manager.update_slots(*update[:-1], operation=update[-1])
                    self.pending_updates.clear()  # 清空更新队列

                    if new_score > best_score:
                        self.update_operation_probabilities(self.operation_type, True)
                        self.best_solution = copy.deepcopy(new_solution)
                        best_score = new_score
                        # if self.verbose:
                        print(f'Iteration {i}, \033[92mnew best solution for {self.seed} found with score: {best_score:.5e}\033[0m')

                        # 记录成功操作导致分数上升的次数
                        self.operation_success_counts[self.operation_type] += 1
                else:
                    self.update_operation_probabilities(self.operation_type, False)
                    self.print(f'Iteration {i}, \033[91mRejected\033[0m new solution with score: {new_score:.5e}')

                self.current_temp *= self.alpha
                if self.current_temp < self.min_temp:
                    break
            else:
                self.update_operation_probabilities(self.operation_type, False)
                print(f'Iteration {i}, \033[94mNo valid neighbor found.\033[0m')

        # 输出每个操作类型成功导致分数上升的次数
        self.print("\nOperation success counts (leading to score increase):")
        for operation, count in self.operation_success_counts.items():
            self.print(f"{operation}: {count} times")
        self.print(f'Best solution found with score: {best_score:.5e}')
        return self.best_solution, best_score

    def evaluate(self, S:DiffSolution):
        input = DiffInput(is_new=True, step=1, diff_solution=self.current_solution)
        return S.SA_evaluation_function(input)


def get_my_solution(seed, verbose=False):
    # 加载数据
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')

    # 初始化插槽可用性管理器
    slot_manager = SlotAvailabilityManager(datacenters)

    # 初始化模拟退火算法
    initial_solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'quantity'])
    initial_solution['dismiss'] = pd.Series(dtype='int')
    S = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(initial_solution, slot_manager=slot_manager, seed=seed, initial_temp=100000, min_temp=100, alpha=0.99, max_iter=500, verbose=verbose)

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

    return

# 主程序
if __name__ == '__main__':
    seed = 3329
    get_my_solution(seed, verbose=True)