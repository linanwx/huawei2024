import math
import numpy as np
import pandas as pd
import random
import copy
import threading

from evaluation_diff_quantity import DiffInput, DiffSolution
from utils import save_solution

TIME_STEPS = 168
DISMISS_MAX = 10000


def generate_new_time_step(old_time_step, TIME_STEPS):
    min_shift = max(-50, 1 - old_time_step)  # 确保不小于1
    max_shift = min(50, TIME_STEPS - old_time_step)  # 确保不大于TIME_STEPS

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

    def __init__(self, datacenters, verbose=False):
        self.slots_table = pd.DataFrame({'time_step': np.arange(1, TIME_STEPS + 1)})
        self.verbose = verbose
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
        available_slots = self.slots_table.loc[start_time - 1:end_time - 1, data_center].min()
        return available_slots

    def update_slots(self, start_time, end_time, data_center, slots_needed, operation='buy'):
        end_time = min(end_time, TIME_STEPS)
        for ts in range(start_time, end_time + 1):
            if operation == 'buy':
                self.slots_table.at[ts - 1, data_center] -= slots_needed
            elif operation == 'cancel':
                self.slots_table.at[ts - 1, data_center] += slots_needed
        if operation == 'buy':
            self._print(f"购买了 {slots_needed * (end_time - start_time)} 个插槽，时间段 {start_time} - {end_time}, 数据中心 {data_center}")
        elif operation == 'cancel':
            self._print(f"取消了 {slots_needed * (end_time - start_time)} 个插槽，时间段 {start_time} - {end_time}, 数据中心 {data_center}")


class SimulatedAnnealing:
    def __init__(self, initial_solution, slot_manager, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
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

        self.operation_probabilities = {
            'buy': 4,
            'adjust_dismiss_time': 1,
            'adjust_time_slot': 0,
            'replace_server': 0,
            "adjust": 0
        }
        self.low_pass_filter_alpha = 0.99
        self.initial_operation_probabilities = copy.deepcopy(self.operation_probabilities)

        self.step_count = int(math.log(min_temp / initial_temp) / math.log(alpha))
        self._print(f"将在 {self.step_count} 步中完成退火过程")

        self.operation_success_counts = {op: 0 for op in self.operation_probabilities.keys()}

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

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
        self.pending_updates = []  

        if self.operation_type == 'buy':
            self._print("临域生成：进行buy操作")
            solution, errinfo = self.generate_new_purchase(current_solution, servers, datacenters)
        elif self.operation_type == 'adjust_dismiss_time':
            self._print("临域生成：进行dismiss时间调整操作")
            solution, errinfo = self.adjust_dismiss_time(current_solution, servers)
        else:
            solution, errinfo = None, "无效的操作"
            self._print(errinfo)
        
        return solution

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

        max_available_slots = self.slot_manager.get_maximum_available_slots(time_step, time_step + life_expectancy, data_center)

        if max_available_slots >= slots_size:
            max_purchase_count = max_available_slots // slots_size // 4  
            purchase_count = random.randint(1, max(1, max_purchase_count))
            total_slots_needed = purchase_count * slots_size
            ID = self.id_gen.next_id()
            new_rows = [{
                'time_step': time_step,
                'datacenter_id': data_center,
                'server_generation': server_generation,
                'server_id': ID,
                'action': 'buy',
                'quantity': purchase_count,
                'dismiss': DISMISS_MAX,
            }]
            new_solution = pd.concat([current_solution, pd.DataFrame(new_rows)], ignore_index=True)
            self.pending_updates = [(time_step, time_step + life_expectancy, data_center, total_slots_needed, 'buy')]
            self._print(f"购买了 {purchase_count} 台 {server_generation} 服务器, 在数据中心 {data_center} 服务器 {ID}，时间段 {time_step} - {time_step + life_expectancy}, 数据中心 {data_center}")
            return new_solution, None
        else:
            return None, f"剩余插槽数量无法选购当前种类服务器，当前需要{slots_size}"

    def adjust_dismiss_time(self, current_solution, servers):
        if current_solution.empty:
            return None, "空解"

        # 深拷贝当前的解，防止修改原解
        new_solution = current_solution.copy(deep=True)

        # 随机选择一个购买操作
        buy_actions = new_solution[new_solution['action'] == 'buy']
        if buy_actions.empty:
            return None, "无可用购买操作步骤"

        # 随机选择一个购买操作
        selected_action_index = random.choice(buy_actions.index)
        selected_action = new_solution.loc[selected_action_index]

        buy_time_step = selected_action['time_step']
        server_generation = selected_action['server_generation']
        life_expectancy = servers.loc[servers['server_generation'] == server_generation, 'life_expectancy'].values[0]

        # 获取当前的 dismiss 时间，如果它未被修改过，使用购买时间步 + life_expectancy
        current_dismiss_time = selected_action['dismiss'] if selected_action['dismiss'] != DISMISS_MAX else buy_time_step + life_expectancy
        current_dismiss_time = min(current_dismiss_time, TIME_STEPS + 1)

        # 计算最小截止时间
        min_dismiss_time = max(current_dismiss_time - TIME_STEPS // 5, buy_time_step)

        if min_dismiss_time >= current_dismiss_time:
            return None, "无法调整截止时间, 最小截止时间大于等于当前截止时间"

        # 在[min_dismiss_time, current_dismiss_time]范围内随机生成新的截止时间
        new_dismiss_time = random.randint(min_dismiss_time, current_dismiss_time)

        # 更新选定的解的 dismiss 时间
        new_solution.at[selected_action_index, 'dismiss'] = new_dismiss_time

        slots_size = servers.loc[servers['server_generation'] == server_generation, 'slots_size'].values[0]
        total_slots_needed = selected_action['quantity'] * slots_size

        # 如果新的 dismiss 时间比当前 dismiss 时间小，则需要释放插槽
        if new_dismiss_time < current_dismiss_time:
            self.pending_updates.append((new_dismiss_time + 1, current_dismiss_time, selected_action['datacenter_id'], total_slots_needed, 'cancel'))

        self._print(f"调整了购买操作的截止时间: 服务器 {selected_action['server_id']}, 购买时间 {buy_time_step}, 原截止时间{current_dismiss_time} 新的截止时间: {new_dismiss_time}")

        return new_solution, None


    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def run(self, S, servers, datacenters):
        current_score = self.evaluate(S)
        best_score = current_score

        for i in range(self.max_iter):
            self._print(f"<------------- Iteration {i}/{self.step_count} ------------->")
            self.current_step = i
            new_solution = self.generate_neighbor(self.current_solution, servers, datacenters)
            if new_solution is not None:
                new_input = DiffInput(is_new=True, step=1, diff_solution=new_solution)
                new_score = S.SA_evaluation_function(new_input)
                accept_prob = self.acceptance_probability(current_score, new_score)

                if accept_prob != 1.0:
                    self._print(f'Acceptance Probability: {accept_prob:.2f}')

                if accept_prob == 1.0 or random.random() < accept_prob:
                    self.current_solution = new_solution
                    current_score = new_score
                    self._print(f'\033[93mAccepted\033[0m new solution with score: {current_score:.5e}')

                    for update in self.pending_updates:
                        self.slot_manager.update_slots(*update[:-1], operation=update[-1])
                    self.pending_updates.clear()

                    if new_score > best_score:
                        self.update_operation_probabilities(self.operation_type, True)
                        self.best_solution = copy.deepcopy(new_solution)
                        best_score = new_score
                        print(f'\033[92mnew best solution for {self.seed} found with score: {best_score:.5e}\033[0m')

                        self.operation_success_counts[self.operation_type] += 1
                else:
                    self.update_operation_probabilities(self.operation_type, False)
                    self._print(f'\033[91mRejected\033[0m new solution with score: {new_score:.5e}')

                self.current_temp *= self.alpha
                if self.current_temp < self.min_temp:
                    break
            else:
                self.update_operation_probabilities(self.operation_type, False)
                self._print(f'\033[94mNo valid neighbor found.\033[0m')

        self._print("\nOperation success counts (leading to score increase):")
        for operation, count in self.operation_success_counts.items():
            self._print(f"{operation}: {count} times")
        self._print(f'Best solution found with score: {best_score:.5e}')
        return self.best_solution, best_score

    def evaluate(self, S: DiffSolution):
        input = DiffInput(is_new=True, step=1, diff_solution=self.current_solution)
        return S.SA_evaluation_function(input)


def get_my_solution(seed, verbose=False):
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')

    slot_manager = SlotAvailabilityManager(datacenters, verbose=verbose)

    initial_solution = pd.DataFrame(columns=['time_step', 'datacenter_id', 'server_generation', 'server_id', 'action', 'quantity'])
    initial_solution['dismiss'] = pd.Series(dtype='int')
    
    S = DiffSolution(seed=seed)
    sa = SimulatedAnnealing(initial_solution, slot_manager=slot_manager, seed=seed, initial_temp=100000, min_temp=100, alpha=0.99, max_iter=500, verbose=verbose)

    best_solution, best_score = sa.run(S, servers, datacenters)

    expanded_rows = []
    dismiss_operations = []

    for _, row in best_solution.iterrows():
        for i in range(row['quantity']):
            new_row = row.drop('quantity').copy()
            new_row['server_id'] = f"{row['server_id']}:{i + 1}"
            expanded_rows.append(new_row)

            # 如果dismiss时间小于等于TIME_STEPS，创建dismiss操作
            if row['dismiss'] <= TIME_STEPS:
                dismiss_row = new_row.copy()
                dismiss_row['action'] = 'dismiss'
                dismiss_row['time_step'] = row['dismiss']
                dismiss_operations.append(dismiss_row)

    expanded_df = pd.DataFrame(expanded_rows)
    dismiss_df = pd.DataFrame(dismiss_operations)

    # 将buy和dismiss操作合并
    final_df = pd.concat([expanded_df, dismiss_df], ignore_index=True)
    final_df = final_df.drop(columns=['dismiss'], errors='ignore')

    save_solution(final_df, f'./output/{seed}_{best_score:.5e}.json')
    save_solution(best_solution, f'./output/quantity_{seed}_{best_score:.5e}.json')
    print(f'Final best solution: {best_solution}')

    return

if __name__ == '__main__':
    seed = 3329
    get_my_solution(seed, verbose=True)
