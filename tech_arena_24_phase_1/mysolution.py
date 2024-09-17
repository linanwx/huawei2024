import numpy as np
import pandas as pd
from seeds import known_seeds
from real_diff_SA import get_my_solution
from multiprocessing import Pool

# 获取训练的 seeds
seeds = known_seeds('test')

extended_seeds = seeds * 4  # 将 seeds 列表重复

# 读取需求数据
demand = pd.read_csv('./data/demand.csv')

# 定义工作函数，将每个进程执行的逻辑封装在函数中
def run_solution(seed):
    # 设置随机种子
    np.random.seed(seed)

    # 调用你的方法
    solution = get_my_solution(seed)

    return solution

if __name__ == '__main__':
    # 创建进程池（根据你的系统核心数调整 processes 的数量，比如 4 或 8）
    with Pool(processes=8) as pool:  # 4 代表创建4个并行进程，或者你可以根据CPU核数设置
        # map函数会将seeds列表中的每个种子分配到不同进程并行执行
        results = pool.imap_unordered(run_solution, extended_seeds)
        for result in pool.imap_unordered(run_solution, extended_seeds):
            # 处理每个任务结果（此处可以输出日志或者其他处理）
            print("任务完成，保存结果")


