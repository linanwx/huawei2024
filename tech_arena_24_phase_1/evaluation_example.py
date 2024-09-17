import time
from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function


# LOAD SOLUTION
# solution = load_solution('./data/solution_example_repeat_org.json')
solution = load_solution('./output/3329_9.76461e+08.json')

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices = load_problem_data()

# START TIMER
start_time = time.time()

# EVALUATE THE SOLUTION
score = evaluation_function(solution,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            seed=3329, verbose = 0)

# END TIMER
end_time = time.time()

# CALCULATE ELAPSED TIME
elapsed_time = end_time - start_time

print(f'Solution score: {score}')
print(f'Elapsed time: {elapsed_time} seconds')