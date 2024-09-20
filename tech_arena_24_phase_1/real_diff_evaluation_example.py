import time
import pandas as pd
from real_diff_evaluation import DiffSolution, ServerMoveInfo, ServerInfo

def load_and_process_operations(file_path):
    """
    读取JSON文件，并处理buy、move和dismiss操作。
    返回一个包含每个服务器生命周期信息的字典。
    """
    try:
        # 读取JSON文件，并指定数据类型
        operations_df = pd.read_json(file_path, dtype={'time_step': int, 'server_id': str})
    except ValueError:
        print(f"文件 {file_path} 不是有效的 JSON 格式。")
        return {}
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return {}

    # 初始化一个存储所有服务器生命周期信息的字典
    server_map = {}

    # 逐行处理每个操作
    for _, row in operations_df.iterrows():
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

    return server_map


# 示例调用
json_file_path = './output/2281_1.10388e+09.json'
server_map = load_and_process_operations(json_file_path)

start_time = time.time()  # 记录开始时间
S = DiffSolution(2281, False)

# 逐个服务器处理，并进行评估
for server_info in server_map.values():
    S.apply_server_change(server_info)
    
score = S.diff_evaluation()
S.commit_server_changes()
# S.commit_server_changes()
print(f'score: {score}')
end_time = time.time()  # 记录结束时间
print(f'运行时间：{end_time - start_time:.2f} 秒')