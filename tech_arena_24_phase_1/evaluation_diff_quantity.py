# 使用差分评估 重新实现evaluation.py中的评估函数
# 当前版本先优化评估函数的性能，后续版本再考虑实现差分的计算SA迭代过程中解变动导致的对应评估值的变化

from collections import defaultdict
import time
import numpy as np
import pandas as pd
from os.path import abspath, join
from dataclasses import dataclass
from scipy.stats import truncweibull_min

from evaluation import get_actual_demand

class DynamicCapacityTracker:
    def __init__(self):
        self.server_generation_set = defaultdict(set)  # key: server_generation, value: set of latency_sensitivity
        self.latency_sensitivity_set = defaultdict(set)  # key: latency_sensitivity, value: set of server_generation
        self.capacity_table = pd.DataFrame()

    def add_server(self, gen, lat, capacity):
        # 如果 capacity_table 是空的，初始化至少一列或一行
        if self.capacity_table.empty:
            self.capacity_table = pd.DataFrame(0, index=[gen], columns=[lat])
        else:
            # 检查并添加新的列
            if lat not in self.capacity_table.columns:
                self.capacity_table[lat] = 0

            # 检查并添加新的行
            if gen not in self.capacity_table.index:
                self.capacity_table.loc[gen] = [0] * len(self.capacity_table.columns)

        # 更新容量
        self.capacity_table.loc[gen, lat] += capacity

        # 更新集合，记录该服务器及其延迟敏感度
        self.server_generation_set[gen].add(lat)
        self.latency_sensitivity_set[lat].add(gen)

    def remove_server(self, gen, lat, capacity):
        # 移除服务器，更新容量表
        self.capacity_table.loc[gen, lat] -= capacity
        # 如果容量为0，删除连接
        if self.capacity_table.loc[gen, lat] == 0:
            self.server_generation_set[gen].discard(lat)
            self.latency_sensitivity_set[lat].discard(gen)
        # 如果没有剩余连接，删除节点
        if not self.server_generation_set[gen]:
            self.capacity_table.drop(gen, axis=0, inplace=True)
        if not self.latency_sensitivity_set[lat]:
            self.capacity_table.drop(lat, axis=1, inplace=True)

    def get_capacity_table(self):
        return self.capacity_table

TIME_STEPS = 168 

@dataclass
class DiffInput:
    is_new: bool                                # 是否只是评估解，而非差分计算
    step: int                                   # 差分时间点
    diff_solution: pd.DataFrame                 # 差分操作

@dataclass
class FleetInfo:
    """
    服役服务器信息

    Attributes:
        fleet (pd.DataFrame): 服役服务器信息
        expiration_map (dict): 服役服务器过期时间映射
        purchase_cost (float, optional): 购买成本，在每个计算时间步骤重置为0
    """
    fleet: pd.DataFrame
    expiration_map: dict
    total_capacity_table: DynamicCapacityTracker = None
    
def get_time_step_solution(solution:pd.DataFrame, ts):
    if ts in solution.index:
        s = solution.loc[[ts]]
        s = s.set_index('server_id', drop=False, inplace=False)
        return s
    else:
        return pd.DataFrame()
    
def get_time_step_demand(demand, ts):
    d = demand[demand['time_step'] == ts]
    d = d.set_index('server_generation', drop=True, inplace=False)
    d = d.drop(columns='time_step', inplace=False)
    return d

def get_utilization(D, Z):
    # CALCULATE OBJECTIVE U = UTILIZATION
    u = []
    server_generations = Z.index
    latency_sensitivities = Z.columns
    for server_generation in server_generations:
        for latency_sensitivity in latency_sensitivities:
            z_ig = Z[latency_sensitivity].get(server_generation, default=0)
            d_ig = D[latency_sensitivity].get(server_generation, default=0)
            if (z_ig > 0) and (d_ig > 0):
                u.append(min(z_ig, d_ig) / z_ig)
            elif (z_ig == 0) and (d_ig == 0):
                continue
            elif (z_ig > 0) and (d_ig == 0):
                u.append(0)
            elif (z_ig == 0) and (d_ig > 0):
                continue
    if u:
        return sum(u) / len(u)
    else:
        return 0

# 重点优化函数 
def get_capacity_by_server_generation_latency_sensitivity(fleet:pd.DataFrame):
    fleet['total_capacity'] = fleet['capacity'] * fleet['quantity']
    Z = fleet.groupby(by=['server_generation', 'latency_sensitivity',], observed=True)['total_capacity'].sum().unstack()
    Z = Z.map(adjust_capacity_by_failure_rate, na_action='ignore')
    Z = Z.fillna(0, inplace=False)
    return Z

def get_normalized_lifespan(fleet):
    # CALCULATE OBJECTIVE L = NORMALIZED LIFESPAN
    return ((fleet['lifespan'] / fleet['life_expectancy']) * fleet['quantity']).sum() / fleet['quantity'].sum()


def adjust_capacity_by_failure_rate(x):
    # HELPER FUNCTION TO CALCULATE THE FAILURE RATE f
    return int(x * (1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item()))

def adjust_capacity_by_failure_rate_numpy(x):
    # 使用 NumPy 矢量化操作来计算故障率
    failure_rate = truncweibull_min.rvs(0.3, 0.05, 0.1, size=x.shape)
    return (x * (1 - failure_rate)).astype(int)

def get_capacity_by_server_generation_latency_sensitivity_optimized(total_capacity_table: DynamicCapacityTracker):
    Z = total_capacity_table.capacity_table.copy()

    # 使用原来的方式调整容量
    Z = adjust_capacity_by_failure_rate_numpy(Z.values)
    return pd.DataFrame(Z, index=total_capacity_table.capacity_table.index, columns=total_capacity_table.capacity_table.columns)

def get_revenue(D, Z, selling_prices):
    # CALCULATE THE REVENUE
    r = 0
    server_generations = Z.index
    latency_sensitivities = Z.columns
    for server_generation in server_generations:
        for latency_sensitivity in latency_sensitivities:
            z_ig = Z[latency_sensitivity].get(server_generation, default=0)
            d_ig = D[latency_sensitivity].get(server_generation, default=0)
            p_ig = selling_prices[latency_sensitivity].get(server_generation, default=0)
            r += min(z_ig, d_ig) * p_ig
    return r

def get_cost(fleet_info:FleetInfo):
    fleet = fleet_info.fleet
    
    # 预计算能源成本，考虑数量
    energy_cost = fleet['energy_consumption'] * fleet['cost_of_energy'] * fleet['quantity']
    
    # 计算维护成本，考虑数量
    x = fleet['lifespan']
    xhat = fleet['life_expectancy']
    b = fleet['average_maintenance_fee']
    ratio = (1.5 * x) / xhat
    maintenance_cost = b * (1 + ratio * np.log2(ratio)) * fleet['quantity']
    
    # 计算总成本，考虑数量
    cost = energy_cost + maintenance_cost
    
    # 计算购买成本，考虑数量
    cost += np.where(x == 1, fleet['purchase_price'] * fleet['quantity'], 0)
    
    # 添加移动成本，考虑数量
    # cost += np.where(fleet['moved'] == 1, fleet['cost_of_moving'] * fleet['quantity'], 0)
    
    return cost.sum()

def get_profit(D, Z, selling_prices, fleet_info):
    # CALCULATE OBJECTIVE P = PROFIT
    R = get_revenue(D, Z, selling_prices)
    C = get_cost(fleet_info)
    return R - C

def change_selling_prices_format(selling_prices):
    # ADJUST THE FORMAT OF THE SELLING PRICES DATAFRAME TO GET ALONG WITH THE
    # REST OF CODE
    selling_prices = selling_prices.pivot(index='server_generation', columns='latency_sensitivity')
    selling_prices.columns = selling_prices.columns.droplevel(0)
    return selling_prices

class DiffSolution:
    # 四个来自文本的csv数据
    demand: pd.DataFrame
    datacenters: pd.DataFrame
    servers: pd.DataFrame
    selling_prices: pd.DataFrame
    selling_prices_cache: pd.DataFrame
    # 缓存数据
    seed_state = None                           # 在生成demand后的种子，用于评估函数
    map_demand_cache: dict = None               # 缓存每个时间步骤的需求

    def __init__(self,seed=None ,path=None):
        np.random.seed(seed)
        # 初始化工作，先加载数据
        if path is None:
            path = './data/'
        self.load_data(path)
        self.demand = get_actual_demand(self.demand) # 根据seed获取修正后的demand
        self.seed_state = np.random.get_state()      # 生成demand后固定种子
        self.map_demand_cache = self.cache_time_step_demand(self.demand) # 缓存每个时间步骤的需求
        self.selling_prices_cache = change_selling_prices_format(self.selling_prices)

    def cache_time_step_demand(self, demand:pd.DataFrame):
        cache = {}
        for ts in range(1, TIME_STEPS + 1):
            cache[ts] = get_time_step_demand(demand, ts)
        return cache

    def load_data(self, path):
        p = abspath(join(path, 'demand.csv'))
        self.demand = pd.read_csv(p)    
        self.demand['latency_sensitivity'] = self.demand['latency_sensitivity'].astype('category')

        # LOAD DATACENTERS DATA
        p = abspath(join(path, 'datacenters.csv'))
        self.datacenters = pd.read_csv(p)
        self.datacenters['datacenter_id'] = self.datacenters['datacenter_id'].astype('category')
        self.datacenters['latency_sensitivity'] = self.datacenters['latency_sensitivity'].astype('category')
        
        # LOAD SERVERS DATA
        p = abspath(join(path, 'servers.csv'))
        self.servers = pd.read_csv(p)
        self.servers['server_generation'] = self.servers['server_generation'].astype('category')
        self.servers['server_type'] = self.servers['server_type'].astype('category')
        self.servers.drop(columns=['release_time'], inplace=True)
        
        # LOAD SELLING PRICES DATA
        p = abspath(join(path, 'selling_prices.csv'))
        self.selling_prices = pd.read_csv(p)
        self.selling_prices['server_generation'] = self.selling_prices['server_generation'].astype('category')
        self.selling_prices['latency_sensitivity'] = self.selling_prices['latency_sensitivity'].astype('category')

    def update_fleet(self, ts: int, fleet_info:FleetInfo, ts_solution:pd.DataFrame) -> FleetInfo:
        fleet = fleet_info.fleet
        capacity_tracker = fleet_info.total_capacity_table  # 动态容量追踪器

        # if ts == 44:
        #     print("fleet")
        #     print(fleet)
        #     # 检查每列类型
        #     print("fleet.dtypes")
        #     print(fleet.dtypes)

        if not ts_solution.empty:
            buy_actions = ts_solution[ts_solution['action'] == 'buy']
            
            if not buy_actions.empty:
                if capacity_tracker is None:
                    capacity_tracker = DynamicCapacityTracker()

                # 动态添加出现的 server_generation 和 latency_sensitivity
                for _, row in buy_actions.iterrows():
                    gen = row['server_generation']
                    lat = row['latency_sensitivity']
                    capacity = row['capacity'] * row['quantity']

                    capacity_tracker.add_server(gen, lat, capacity)

                fleet = pd.concat([fleet, buy_actions], ignore_index=False)

                # 计算并累加购买成本
                # fleet_info.purchase_cost = buy_actions['purchase_price'].sum()

                # 提前计算服务器的到期时间，方便后面删除过期服务器
                life_expectancies = buy_actions['life_expectancy'] + ts
                for server_id, expire_ts in zip(buy_actions['server_id'], life_expectancies):
                    if expire_ts in fleet_info.expiration_map:
                        fleet_info.expiration_map[expire_ts].append(server_id)
                    else:
                        fleet_info.expiration_map[expire_ts] = [server_id]

            # move_actions = ts_fleet[ts_fleet['action'] == 'move']
            # if not move_actions.empty:
            #     move_indices = move_actions['server_id']
            #     fleet.loc[move_indices, 'datacenter_id'] = move_actions['datacenter_id']
            #     fleet.loc[move_indices, 'moved'] = 1

            # dismiss_actions = ts_fleet[ts_fleet['action'] == 'dismiss']
            # if not dismiss_actions.empty:
            #     dismiss_indices = dismiss_actions['server_id']
            #     fleet.drop(index=dismiss_indices, inplace=True)

        fleet['lifespan'] += 1

        # 删除过期的服务器（如果有）
        if ts in fleet_info.expiration_map:
            expire_indices = fleet_info.expiration_map.pop(ts)
            mask = ~fleet.index.isin(expire_indices)
            expired_fleet = fleet[~mask]
            for _, row in expired_fleet.iterrows():
                gen = row['server_generation']
                lat = row['latency_sensitivity']
                capacity = row['capacity'] * row['quantity']

                # 使用容量追踪器动态删除服务器容量
                capacity_tracker.remove_server(gen, lat, capacity)
            fleet = fleet[mask]

        fleet_info.fleet = fleet
        fleet_info.total_capacity_table = capacity_tracker
        return fleet_info

    def solution_data_preparation(self, solution:pd.DataFrame):
        # 为解准备数据
        # 1. 为解添加服务器的数据
        solution = solution.merge(self.servers, how='left', on='server_generation')
        # 2. 为解添加数据中心的数据
        solution = solution.merge(self.datacenters, how='left', on='datacenter_id')
        solution = solution.drop(columns=['slots_capacity'])
        # 3. 为解添加售价数据
        solution = solution.merge(self.selling_prices, how='left', on=['server_generation', 'latency_sensitivity'])

        solution["lifespan"] = pd.Series([0] * len(solution), dtype=int)
        solution.set_index('time_step', inplace=True)
        # print("solution.dtypes")
        # print(solution.dtypes)
        # print("solution")
        # print(solution)
        return solution

    def SA_evaluation_function(self, diff_input:DiffInput, time_steps=TIME_STEPS):
        start_time = time.time()
        # 重设种子
        np.random.set_state(self.seed_state)
        # 初始化
        start = 1
        df_ULP : pd.DataFrame = pd.DataFrame({
            'time-step': np.arange(1, time_steps + 1),
            'U': np.zeros(time_steps),
            'L': np.zeros(time_steps),
            'P': np.zeros(time_steps)
        })

        # 先不差分评估，先全部计算
        solution:pd.DataFrame = diff_input.diff_solution
        # 连接每个步骤的费用数据
        solution = self.solution_data_preparation(solution)

        # 初始化 fleet 信息 fleet使用和解相同的布局，设置server_id为index
        column_dtypes = solution.dtypes.to_dict()
        fleet = pd.DataFrame(columns=solution.columns).astype(column_dtypes)
        fleet.set_index(['server_id'], drop=False, inplace=True)

        fleet_info:FleetInfo = FleetInfo(fleet=fleet, expiration_map={})

        for ts in range(start, time_steps + 1):
            # 重置循环内的数据
            # fleet_info.purchase_cost = 0.0
            # 从缓存获取当前步骤需求
            D = self.map_demand_cache[ts]
            # 从解获取当前步骤操作
            ts_fleet = get_time_step_solution(solution, ts)

            if ts_fleet.empty and fleet_info.fleet.empty:
                continue
            # 根据当前步骤操作，差分更新服役服务器 ！！！超级耗时！！！
            fleet_info = self.update_fleet(ts, fleet_info, ts_fleet)

            if fleet_info.fleet.shape[0] > 0:
                # 计算调整后容量 ！！！超级耗时！！！
                Zf = get_capacity_by_server_generation_latency_sensitivity_optimized(fleet_info.total_capacity_table)

                # if ts == 7:
                #     print(Zf)
                #     exit()

                U = get_utilization(D, Zf)

                # print(U)
                # exit()

                L = get_normalized_lifespan(fleet_info.fleet)

                P = get_profit(D, Zf, self.selling_prices_cache, fleet_info)

                # 记录当前步骤的U, L, P
                df_ULP.at[ts-1, 'U'] = U
                df_ULP.at[ts-1, 'L'] = L
                df_ULP.at[ts-1, 'P'] = P

                # PREPARE OUTPUT
                output = {'time-step': ts,
                    'U': round(U, 2),
                    'L': round(L, 2),
                    'P': round(P, 2),
                }
            
                # print(output)
                # print(fleet_info.fleet)
            else:
                print("No fleet information available.")
        total_value = (df_ULP['U'] * df_ULP['L'] * df_ULP['P']).sum()
        end_time = time.time()

        print(f"Time SA_evaluation_function taken: {end_time - start_time} seconds")

        return total_value
