import numpy as np
import pandas as pd
import math
import random
from copy import deepcopy
import logging

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
        self.data = self.initialize_data()
    
    def initialize_data(self):
        return {
            'servers': [],  # 服务器列表
            'datacenters': {'dc1': 25245, 'dc2': 15300, 'dc3': 7020, 'dc4': 8280},  # 每个数据中心的初始容量
            'operations': []  # 操作历史
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
    
    def _check_buy_constraints(self):
        total_capacity_used = sum(self.data['datacenters'].values())
        max_capacity = 1000  # 数据中心总容量限制

        # 检查是否有足够的容量购买新的服务器
        if total_capacity_used >= max_capacity:
            logging.info("Buy constraint failed: Data centers are full.")
            return False

        # 检查服务器数量限制
        max_servers = 50  # 服务器数量上限
        if len(self.data['servers']) >= max_servers:
            logging.info("Buy constraint failed: Server limit reached.")
            return False

        return True
    
    def _check_dismiss_constraints(self):
        # 检查是否有操作历史可以撤销
        if len(self.history) == 0:
            logging.info("Dismiss constraint failed: No operations to dismiss.")
            return False

        return True
    
    def _check_move_constraints(self):
        if len(self.data['servers']) == 0:
            logging.info("Move constraint failed: No servers to move.")
            return False

        # 假设随机选择一个目标数据中心
        target_datacenter = random.choice(list(self.data['datacenters'].keys()))
        moving_server = random.choice(self.data['servers'])
        available_capacity = self.data['datacenters'][target_datacenter]

        if moving_server['capacity'] > available_capacity:
            logging.info(f"Move constraint failed: Not enough capacity in {target_datacenter}.")
            return False

        return True
    
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
    
    def _apply_buy(self):
        # 假设从CSV加载的服务器生成数据
        server = random.choice(self.server_generation)
        new_server = {'id': len(self.data['servers']) + 1, 'capacity': server['capacity']}
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

# 读取 CSV 文件中的数据
servers_data = pd.read_csv('servers.csv')

# 提取服务器生成、类型和发布时间
server_generation = servers_data['server_generation'].tolist()
server_types = servers_data['server_type'].tolist()
release_times = servers_data['release_time'].tolist()

# 初始化参数
initial_state = {
    'servers': [],  # 初始没有服务器
    'datacenters': {'dc1': 25245, 'dc2': 15300, 'dc3': 7020, 'dc4': 8280},  # 数据中心容量
    'operations': []  # 初始没有操作记录
}
initial_temperature = 1000
cooling_rate = 0.95
num_iterations = 1000

# 创建模拟退火对象并优化
sa = SimulatedAnnealing(objective_function, initial_state, initial_temperature, cooling_rate, num_iterations)
sa.server_generation = [{'capacity': cap} for cap in server_generation]  # 传递服务器生成数据
best_state, best_energy = sa.optimize()

print(f"最佳状态: {best_state}")
print(f"最佳能量: {best_energy}")
