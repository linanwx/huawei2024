import os
import time
from utils import load_problem_data, load_solution
from evaluation import evaluation_function
import multiprocessing as mp

# Function to load all solutions in a directory using load_solution
def load_all_solutions(directory):
    solutions = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file0 = os.path.splitext(filename)[0]
            # 如果file0 是 数字下划线分割
            if '_' in file0:
                seed = int(file0.split('_')[0])
            else:
                seed = int(file0)  # Get the seed from the filename (without .json)
            file_path = os.path.join(directory, filename)
            solution, price = load_solution(file_path)
            solutions[seed] = (solution, price)
    return solutions

# 这是你原本的评估函数
def evaluate_solution_wrapper(args):
    seed, (solution, price), demand, datacenters, servers, selling_prices, p = args
    print(f"Evaluating solution for seed: {seed}")
    start_time = time.time()
    
    # 调用评估函数
    score = evaluation_function(solution, price, demand, datacenters, servers, selling_prices, p, seed=seed, verbose=1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Solution score: {score} for seed {seed}')
    print(f'Elapsed time: {elapsed_time} seconds')
    
    return score

# 多进程版本的evaluate_solutions
def evaluate_solutions(solutions, demand, datacenters, servers, selling_prices, p, num_processes=None):
    # 创建进程池，num_processes为None时默认使用机器的核心数
    with mp.Pool(processes=num_processes) as pool:
        # 构建参数列表，将每个解包装成元组，传递给进程池
        args = [(seed, (solution, price), demand, datacenters, servers, selling_prices, p) for seed, (solution, price) in solutions.items()]
        
        # 使用map方法进行多进程运算
        scores = pool.map(evaluate_solution_wrapper, args)

    return scores

# Function to calculate the average score
def calculate_average_score(scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)

# MAIN FUNCTION
def score_main(directory):
    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices, p = load_problem_data()

    # LOAD ALL SOLUTIONS FROM DIRECTORY
    solutions = load_all_solutions(directory)

    # EVALUATE ALL SOLUTIONS
    scores = evaluate_solutions(solutions, demand, datacenters, servers, selling_prices, p)

    # CALCULATE AVERAGE SCORE
    average_score = calculate_average_score(scores)
    print(f'Average score: {average_score:.5e}')

# Specify the directory where JSON files are located
if __name__ == "__main__":
    directory = './2024-09-22-08-03-33'  # Update this path to your JSON directory
    score_main(directory)
