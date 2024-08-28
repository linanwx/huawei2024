import time
import pandas as pd

from evaluation_diff_quantity import DiffInput, DiffSolution

def load_solution(path):
    # Loads a solution from a json file to a pandas DataFrame.
    df = pd.read_json(path, orient='records')
    
        # 转换合适的列为 category 类型以优化内存使用
    df['datacenter_id'] = df['datacenter_id'].astype('category')
    df['server_generation'] = df['server_generation'].astype('category')
    df['action'] = df['action'].astype('category')
    
    # 如果 server_id 是非常多且唯一的，可以保持 object 类型，如果重复率较高，也可以考虑转换为 category
    # if df['server_id'].nunique() < len(df) / 2:  # 假设重复率超过 50% 时
    #     df['server_id'] = df['server_id'].astype('category')
    
    # 打印优化后的内存使用情况
    # print(f"Optimized Solution DataFrame memory usage:\n{df.memory_usage(deep=True)}")
    
    return df
# LOAD SOLUTION
solution = load_solution('./output/quantity_123_407295596.17477804.json')
# exit()



# LOAD PROBLEM DATA
# demand, datacenters, servers, selling_prices = load_problem_data()

# START TIMER
start_time = time.time()

# EVALUATE THE SOLUTION
# score = evaluation_function(solution,
#                             demand,
#                             datacenters,
#                             servers,
#                             selling_prices,
#                             seed=123, verbose = 1)

S = DiffSolution(seed=123)

input:DiffInput = DiffInput(is_new=True, step=1, diff_solution=solution)

for i in range(1):
    score = S.SA_evaluation_function(input)
    print(f'Solution score: {score}')

# END TIMER
end_time = time.time()

# CALCULATE ELAPSED TIME
elapsed_time = end_time - start_time

# print(f'Solution score: {score}')
print(f'Elapsed time: {elapsed_time} seconds')