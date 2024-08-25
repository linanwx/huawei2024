import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution  # 加载 utils 文件中的 save_solution
from evaluation import get_actual_demand
import math
import random
from copy import deepcopy
import logging

def get_my_solution(demand_data):
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    class SimulatedAnnealing:
        def __init__(self, objective_function, initial_state, initial_temperature, cooling_rate, num_iterations):
            self.objective_function = objective_function
            self.current_state = initial_state
            self.current_energy = self.objective_function(self.current_state)
            self.best_state = self.current_state
            self.best_energy = self.current_energy
            self.temperature = initial_temperature
            self.cooling_rate = cooling_rate
            self.num_iterations = num_iterations
            self.history = []
            self.data = self.initialize_data(demand_data)

        def initialize_data(self, demand_data):
            # 读服务器数据
            servers = pd.read_csv('./data/servers.csv')

            return {
                'servers': servers.to_dict('records'),  # 将服务器数据转化为字典列表
                'datacenters': {'dc1': 25245, 'dc2': 15300, 'dc3': 7020, 'dc4': 8280},  # 容量
                'operations': []
            }

        def _generate_new_state(self):
            step_id = random.choice(['buy', 'dismiss', 'move'])
            logging.info(f"Attempting {step_id} operation.")

            operation_map = {
                'buy': self._execute_buy,
                'dismiss': self._execute_dismiss,
                'move': self._execute_move
            }

            return operation_map[step_id]()

        def _acceptance_probability(self, delta_energy):
            if delta_energy < 0:
                return 1.0
            return math.exp(-delta_energy / self.temperature)

        def _execute_buy(self):
            if self._check_buy_constraints():
                new_state = self._apply_buy()
                self.history.append(('buy', deepcopy(self.data)))
                return new_state
            return self.current_state

        def _execute_dismiss(self):
            if self._check_dismiss_constraints():
                new_state = self._apply_dismiss()
                self.history.append(('dismiss', deepcopy(self.data)))
                return new_state
            return self.current_state

        def _execute_move(self):
            if self._check_move_constraints():
                new_state = self._apply_move()
                self.history.append(('move', deepcopy(self.data)))
                return new_state
            return self.current_state

        def _check_buy_constraints(self):
            return bool(self.data['servers'])

        def _check_dismiss_constraints(self):
            return bool(self.history)

        def _check_move_constraints(self):
            return True

        def _apply_buy(self):
            new_server = {'id': len(self.data['servers']) + 1, 'capacity': random.randint(1, 10)}
            self.data['servers'].append(new_server)
            return deepcopy(self.data)

        def _apply_dismiss(self):
            if self.history:
                last_action, last_data = self.history.pop()
                self.data = last_data
            return deepcopy(self.data)

        def _apply_move(self):
            if self.data['servers']:
                server = self.data['servers'].pop()
                new_datacenter = random.choice(list(self.data['datacenters'].keys()))
                self.data['datacenters'][new_datacenter] += server['capacity']
            return deepcopy(self.data)

        def _rollback(self):
            if not self.history:
                return self.current_state
            last_action, last_state = self.history.pop()
            self.data = last_state
            return deepcopy(self.data)

        def optimize(self):
            iteration = 0
            while iteration < self.num_iterations:
                new_state = self._generate_new_state()
                new_energy = self.objective_function(new_state)
                delta_energy = new_energy - self.current_energy

                if random.random() < self._acceptance_probability(delta_energy):
                    self.current_state = new_state
                    self.current_energy = new_energy

                    if self.current_energy < self.best_energy:
                        self.best_state = self.current_state
                        self.best_energy = self.current_energy
                else:
                    self.current_state = self._rollback()
                    self.current_energy = self.objective_function(self.current_state)

                self.temperature *= self.cooling_rate
                iteration += 1

            return self.best_state, self.best_energy

    # 示例目标函数
    def objective_function(data):
        return sum(server['capacity'] for server in data['servers'])

    # 初始化参数
    initial_state = {
        'servers': [],  # 初始没有服务器
        'datacenters': {'dc1': 25245, 'dc2': 15300, 'dc3': 7020, 'dc4': 8280},  # 数据中心容量
        'operations': []
    }
    initial_temperature = 1000
    cooling_rate = 0.95
    num_iterations = 1000

    # 创建模拟退火对象并优化
    sa = SimulatedAnnealing(objective_function, initial_state, initial_temperature, cooling_rate, num_iterations)
    best_state, best_energy = sa.optimize()

    logging.info(f"最佳状态: {best_state}")
    logging.info(f"最佳能量: {best_energy}")

    # 保存优化结果
    return [best_state]

# 读取种子并进行求解
seeds = known_seeds('training')
demand = pd.read_csv('./data/demand.csv')  # 从CSV文件中读取需求数据

for seed in seeds:
    np.random.seed(seed)
    actual_demand = get_actual_demand(demand)  # 根据需求文件中的内容计算实际需求
    solution = get_my_solution(actual_demand)  # 将需求数据传入优化算法
    save_solution(solution, f'./output/{seed}.json')  # 保存优化后的解决方案
