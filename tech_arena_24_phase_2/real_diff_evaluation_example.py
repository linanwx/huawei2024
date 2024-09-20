import json
import time
import pandas as pd
from real_diff_evaluation import LATENCY_SENSITIVITY_MAP, SERVER_GENERATION_MAP, DiffSolution, ServerMoveInfo, ServerInfo

def load_selling_price_dict(file_path):
    """
    读取 selling_prices.csv 文件，构建 selling_price_dict。
    """
    selling_prices_df = pd.read_csv(file_path)
    selling_prices_df['server_generation'] = selling_prices_df['server_generation'].astype(str)
    selling_prices_df['latency_sensitivity'] = selling_prices_df['latency_sensitivity'].astype(str)
    selling_price_dict = selling_prices_df.set_index(['server_generation', 'latency_sensitivity'])['selling_price'].to_dict()
    return selling_price_dict

def generate_pricing_steps(pricing_map, selling_price_dict):
    """
    生成每个时间步的价格信息。
    pricing_map 是初始价格调整映射，
    selling_price_dict 是默认价格字典。
    """
    # 初始化最终的定价数据结构
    pricing_steps = {}

    # 初始化所有步骤和所有组合的价格，168个时间步
    for time_step in range(1, 169):
        pricing_steps[time_step] = {}
        for server_gen in SERVER_GENERATION_MAP.keys():
            for latency_sens in LATENCY_SENSITIVITY_MAP.keys():
                # 使用默认的价格字典填充每个时间步的价格
                default_price = selling_price_dict.get((server_gen, latency_sens), None)
                pricing_steps[time_step][(server_gen, latency_sens)] = default_price

    # 遍历 pricing_map，将定价信息覆盖到 pricing_steps 中
    last_prices = {}  # 记录每个 server_gen 和 latency_sens 的上一次价格
    for (time_step, server_generation), price_info in pricing_map.items():
        latency_sensitivity = price_info['latency_sensitivity']
        price = price_info['price']
        
        # 直接使用字符串形式的 server_generation 和 latency_sensitivity
        if server_generation is not None and latency_sensitivity is not None:
            # 将该时间步的价格设定为新的价格
            for step in range(time_step, 169):
                pricing_steps[step][(server_generation, latency_sensitivity)] = price
            # 更新上一次价格
            last_prices[(server_generation, latency_sensitivity)] = price

    return pricing_steps



def load_and_process_operations(file_path):
    """
    读取JSON文件，并处理buy、move和dismiss操作。
    返回包含每个服务器生命周期信息和价格策略信息的字典。
    """
    try:
        # 读取JSON文件
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
    pricing_map = {}

    # 逐行处理 fleet 中的每个操作
    for row in fleet_operations:
        server_id = row['server_id']
        action = row['action']
        
        if action == 'buy':
            # 如果是buy操作，创建ServerInfo对象
            move_info = ServerMoveInfo(time_step=row['time_step'] - 1, target_datacenter=row['datacenter_id'])
            server_info = ServerInfo(
                server_id=server_id,
                dismiss_time=168,  # 默认生命周期结束为168
                buy_and_move_info=[move_info],
                quantity=1,  # 假设每次购买的数量是1
                server_generation=row['server_generation']
            )
            # 添加到server_map中
            server_map[server_id] = server_info
        
        elif action == 'move':
            # 如果是move操作，更新对应服务器的move_info
            if server_id in server_map:
                move_info = ServerMoveInfo(time_step=row['time_step'] - 1, target_datacenter=row['datacenter_id'])
                server_map[server_id].buy_and_move_info.append(move_info)
                server_map[server_id].init_buy_and_move_info()
            else:
                print(f"未找到服务器 {server_id} 的购买记录，无法执行move操作。")
        
        elif action == 'dismiss':
            # 如果是dismiss操作，更新对应服务器的dismiss_time
            if server_id in server_map:
                server_map[server_id].dismiss_time = row['time_step'] - 1
            else:
                print(f"未找到服务器 {server_id} 的购买记录，无法执行dismiss操作。")

    # 处理 pricing strategy 信息
    for price_info in pricing_strategy:
        time_step = price_info['time_step']
        server_generation = price_info['server_generation']
        price = price_info['price']
        
        # 存储价格策略信息
        pricing_map[(time_step, server_generation)] = {
            'latency_sensitivity': price_info['latency_sensitivity'],
            'price': price
        }

    return server_map, generate_pricing_steps(pricing_map, load_selling_price_dict('./data/selling_prices.csv'))


# 示例调用
json_file_path = './data/solution_example.json'
server_map, pricing_steps = load_and_process_operations(json_file_path)

# print(f'{server_map}\n, {pricing_steps}')

# for key, value in pricing_steps.items():
#     print(f'key: {key}: value: {value}\n\n')
    # step = key
    # gen, price = value
    # print(f'{step}, {gen}, {price}')

start_time = time.time()  # 记录开始时间
S = DiffSolution(123, True)

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