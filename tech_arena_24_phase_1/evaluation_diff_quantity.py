# 使用差分评估 重新实现evaluation.py中的评估函数
# 当前版本先优化评估函数的性能，后续版本再考虑实现差分的计算SA迭代过程中解变动导致的对应评估值的变化

import time
import numpy as np
import pandas as pd
from os.path import abspath, join
from dataclasses import dataclass
from scipy.stats import truncweibull_min

from evaluation import get_actual_demand

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
    # purchase_cost: float = 0.0


def get_time_step_fleet(solution:pd.DataFrame, ts):
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
def get_capacity_by_server_generation_latency_sensitivity(fleet):
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
    return int(x * 1 - truncweibull_min.rvs(0.3, 0.05, 0.1, size=1).item())

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
    cost += np.where(fleet['moved'] == 1, fleet['cost_of_moving'] * fleet['quantity'], 0)
    
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

    def update_fleet(self, ts: int, fleet_info:FleetInfo, ts_fleet:pd.DataFrame) -> FleetInfo:
        fleet = fleet_info.fleet

        if not ts_fleet.empty:
            buy_actions = ts_fleet[ts_fleet['action'] == 'buy']
            buy_actions["lifespan"] = 0
            buy_actions["moved"] = 0
            
            if not buy_actions.empty:
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

            move_actions = ts_fleet[ts_fleet['action'] == 'move']
            if not move_actions.empty:
                move_indices = move_actions['server_id']
                fleet.loc[move_indices, 'datacenter_id'] = move_actions['datacenter_id']
                fleet.loc[move_indices, 'moved'] = 1

            dismiss_actions = ts_fleet[ts_fleet['action'] == 'dismiss']
            if not dismiss_actions.empty:
                dismiss_indices = dismiss_actions['server_id']
                fleet.drop(index=dismiss_indices, inplace=True)

        fleet['lifespan'] += 1

        # fleet.drop(fleet.index[fleet['lifespan'] >= fleet['life_expectancy']], inplace=True)
        # 使用 expiration_map 来删除过期的服务器
        if ts in fleet_info.expiration_map:
            expire_indices = fleet_info.expiration_map.pop(ts)
            # 检查一下fleet的index
            # print(f"fleet = fleet.drop, fleet index: {fleet.index}")
            # fleet.drop(index=expire_indices, inplace=True)
            mask = ~fleet.index.isin(expire_indices)
            fleet = fleet[mask]

        fleet_info.fleet = fleet
        return fleet_info

    def solution_data_preparation(self, solution:pd.DataFrame):
        # 为解准备数据
        # 1. 为解添加服务器的数据
        solution = solution.merge(self.servers, how='left', on='server_generation')
        # 2. 为解添加数据中心的数据
        solution = solution.merge(self.datacenters, how='left', on='datacenter_id')
        # 3. 为解添加售价数据
        solution = solution.merge(self.selling_prices, how='left', on=['server_generation', 'latency_sensitivity'])
        
        solution.set_index('time_step', inplace=True)

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
        fleet = fleet.set_index(['server_id'])
        fleet_info:FleetInfo = FleetInfo(fleet=fleet, expiration_map={})

        for ts in range(start, time_steps + 1):
            # 重置循环内的数据
            # fleet_info.purchase_cost = 0.0
            # 从缓存获取当前步骤需求
            D = self.map_demand_cache[ts]
            # 从解获取当前步骤操作
            ts_fleet = get_time_step_fleet(solution, ts)

            if ts_fleet.empty and fleet_info.fleet.empty:
                continue
            # 根据当前步骤操作，差分更新服役服务器 ！！！超级耗时！！！
            fleet_info = self.update_fleet(ts, fleet_info, ts_fleet)

            if fleet_info.fleet.shape[0] > 0:
                # 计算调整后容量 ！！！超级耗时！！！
                Zf = get_capacity_by_server_generation_latency_sensitivity(fleet_info.fleet)

                # print(Zf)
                # if ts >= 21:
                #     exit()

                U = get_utilization(D, Zf)

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
                    "Fleet size": fleet_info.fleet.shape[0],
                }
            
                # print(output)
                # print(fleet_info.fleet)
            else:
                print("No fleet information available.")
        total_value = (df_ULP['U'] * df_ULP['L'] * df_ULP['P']).sum()
        end_time = time.time()

        print(f"Time SA_evaluation_function taken: {end_time - start_time} seconds")

        return total_value
