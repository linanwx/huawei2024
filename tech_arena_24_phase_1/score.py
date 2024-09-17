import os
import time
from utils import load_problem_data, load_solution
from evaluation import evaluation_function

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
            solution = load_solution(file_path)
            solutions[seed] = solution
    return solutions

# Function to evaluate each solution and return the score
def evaluate_solutions(solutions, demand, datacenters, servers, selling_prices):
    scores = []
    for seed, solution in solutions.items():
        print(f"Evaluating solution for seed: {seed}")
        start_time = time.time()

        # Evaluate the solution
        score = evaluation_function(solution, demand, datacenters, servers, selling_prices, seed=seed, verbose=1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Solution score: {score} for seed {seed}')
        print(f'Elapsed time: {elapsed_time} seconds')
        scores.append(score)
    return scores

# Function to calculate the average score
def calculate_average_score(scores):
    if not scores:
        return 0
    return sum(scores) / len(scores)

# MAIN FUNCTION
def score_main(directory):
    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    # LOAD ALL SOLUTIONS FROM DIRECTORY
    solutions = load_all_solutions(directory)

    # EVALUATE ALL SOLUTIONS
    scores = evaluate_solutions(solutions, demand, datacenters, servers, selling_prices)

    # CALCULATE AVERAGE SCORE
    average_score = calculate_average_score(scores)
    print(f'Average score: {average_score:.5e}')

# Specify the directory where JSON files are located
if __name__ == "__main__":
    directory = './2024-09-17-23-59-37'  # Update this path to your JSON directory
    score_main(directory)
