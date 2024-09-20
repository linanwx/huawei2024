import json
import time
import pandas as pd
from real_diff_evaluation import LATENCY_SENSITIVITY_MAP, SERVER_GENERATION_MAP, DiffSolution, ServerMoveInfo, ServerInfo, export_solution_to_json

def load_selling_price_dict(file_path):
    """
    读取 selling_prices.csv 文件，构建 selling_price_dict。
    """
    selling_prices_df = pd.read_csv(file_path)
    selling_prices_df['server_generation'] = selling_prices_df['server_generation'].astype(str)
    selling_prices_df['latency_sensitivity'] = selling_prices_df['latency_sensitivity'].astype(str)
    selling_price_dict = selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()
    return selling_price_dict

def generate_pricing_steps(pricing_pd, selling_price_dict):
    """
    生成每个时间步的价格信息。
    pricing_pd 是包含 pricing_strategy 的 Pandas DataFrame。
    selling_price_dict 是默认价格字典。
    """
    # 初始化最终的定价数据结构
    pricing_steps = {}

    # 初始化 last_prices 为空字典，用于保存上一次定价
    last_prices = {}

    # 遍历 168 个时间步
    for time_step in range(1, 169):
        pricing_steps[time_step] = {}

        # 筛选出当前时间步的价格信息
        current_step_df = pricing_pd[pricing_pd['time_step'] == time_step]

        # 遍历每个服务器代数和时延敏感度的组合
        for server_gen in SERVER_GENERATION_MAP.keys():
            for latency_sens in LATENCY_SENSITIVITY_MAP.keys():
                # 查找当前时间步的价格
                current_price_row = current_step_df[
                    (current_step_df['server_generation'] == server_gen) &
                    (current_step_df['latency_sensitivity'] == latency_sens)
                ]

                if not current_price_row.empty:
                    # 如果有当前时间步的价格，使用它
                    current_price = current_price_row['price'].values[0]
                    last_prices[(server_gen, latency_sens)] = current_price  # 更新上一次价格
                else:
                    # 使用上一次价格或者默认价格
                    current_price = last_prices.get(
                        (server_gen, latency_sens),
                        selling_price_dict.get((server_gen, latency_sens), None)
                    )

                # 保存当前时间步的价格
                pricing_steps[time_step][(server_gen, latency_sens)] = current_price

    return pricing_steps

def process_pricing_strategy(pricing_strategy):
    """
    使用 Pandas 表格处理 pricing strategy 信息，返回 pricing_map 字典。
    """
    # 将 pricing_strategy 转换为 DataFrame
    pricing_df = pd.DataFrame(pricing_strategy)
    
    # 设置 'time_step' 和 'server_generation' 为索引，方便之后查询
    pricing_df['server_generation'] = pricing_df['server_generation'].astype(str)
    pricing_df['latency_sensitivity'] = pricing_df['latency_sensitivity'].astype(str)
    
    # 创建字典格式的 pricing_map，按时间步和服务器代数进行分组
    # pricing_map = pricing_df.set_index(['time_step', 'server_generation'])[['latency_sensitivity', 'price']]
    
    return pricing_df

def load_and_process_operations(file_path):
    """
    读取 JSON 文件，并处理 buy、move 和 dismiss 操作。
    返回包含每个服务器生命周期信息和价格策略信息的字典。
    """
    try:
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)
    except ValueError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")
        return {}, {}
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return {}, {}

    # 提取 fleet 操作和 pricing strategy
    fleet_operations = data.get('fleet', [])
    pricing_strategy = data.get('pricing_strategy', [])

    # 初始化存储所有服务器生命周期信息和价格策略的字典
    server_map = {}

    # 逐行处理 fleet 中的每个操作
    for row in fleet_operations:
        server_id = row['server_id']
        action = row['action']
        
        if action == 'buy':
            # 如果是 buy 操作，创建 ServerInfo 对象
            move_info = ServerMoveInfo(time_step=row['time_step'] - 1, target_datacenter=row['datacenter_id'])
            server_info = ServerInfo(
                server_id=server_id,
                dismiss_time=168,  # 默认生命周期结束为 168
                buy_and_move_info=[move_info],
                quantity=1,  # 假设每次购买的数量是 1
                server_generation=row['server_generation']
            )
            # 添加到 server_map 中
            server_map[server_id] = server_info
        
        elif action == 'move':
            # 如果是 move 操作，更新对应服务器的 move_info
            if server_id in server_map:
                move_info = ServerMoveInfo(time_step=row['time_step'] - 1, target_datacenter=row['datacenter_id'])
                server_map[server_id].buy_and_move_info.append(move_info)
                server_map[server_id].init_buy_and_move_info()
            else:
                print(f"未找到服务器 {server_id} 的购买记录，无法执行 move 操作。")
        
        elif action == 'dismiss':
            # 如果是 dismiss 操作，更新对应服务器的 dismiss_time
            if server_id in server_map:
                server_map[server_id].dismiss_time = row['time_step'] - 1
            else:
                print(f"未找到服务器 {server_id} 的购买记录，无法执行 dismiss 操作。")

    # 使用 Pandas 处理 pricing strategy 信息
    pricing_df = process_pricing_strategy(pricing_strategy)

    # 读取默认的 selling price
    default_selling_price_dict = load_selling_price_dict('./data/selling_prices.csv')
    
    # 生成每一步的定价
    each_step_price = generate_pricing_steps(pricing_df, default_selling_price_dict)

    return server_map, each_step_price


# 示例调用
json_file_path = './output/3329_1.51956e+09.json'
server_map, pricing_steps = load_and_process_operations(json_file_path)


# print(f'{server_map}\n, {pricing_steps}')

# for key, value in pricing_steps.items():
#     print(f'key: {key}: value: {value}\n\n')
    # step = key
    # gen, price = value
    # print(f'{step}, {gen}, {price}')

start_time = time.time()  # 记录开始时间
S = DiffSolution(3329, True)

# 逐个服务器处理，并进行评估
for server_info in server_map.values():
    S.apply_server_change(server_info)

orginal_price = load_selling_price_dict('./data/selling_prices.csv')

for price_step, price_info in pricing_steps.items():
    for key, value in price_info.items():
        server_gen, latency_sensitivity = key
        price = value
        ratio = 1.0 * price/ orginal_price.get((server_gen, latency_sensitivity), None)
        S.adjust_price_ratio(price_step - 1, price_step , latency_sensitivity, server_gen, ratio)
    
score = S.diff_evaluation()
S.commit_server_changes()
# S.commit_server_changes()
print(f'score: {score}')
end_time = time.time()  # 记录结束时间
print(f'运行时间：{end_time - start_time:.2f} 秒')