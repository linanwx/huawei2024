

from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function


# LOAD SOLUTION
fleet, pricing_strategy = load_solution('./output/3329_1.45831e+09.json')

# LOAD PROBLEM DATA
demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

# EVALUATE THE SOLUTION
score = evaluation_function(fleet, 
                            pricing_strategy,
                            demand,
                            datacenters,
                            servers,
                            selling_prices,
                            elasticity,
                            seed=3329, verbose=True)

print(f'Solution score: {score}')



