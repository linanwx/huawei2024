# Server Cluster Management Optimization

This repository contains our solution for the Huawei Ireland Research Center's Tech Arena 2024 competition, where we achieved first place in the algorithm section of the server cluster management model problem, significantly outperforming the second place. However, due to a lack of emphasis on the PPT and presentation, we finished third overall in the final competition.


## Problem Description

### Problem 1: Server Cluster Management Model

The objective is to recommend the types and quantities of servers to deploy at each time step to maximize profit. Participants manage two types of servers (CPU and GPU) across four data centers. Actions include purchasing, moving, maintaining, and dismissing servers, as well as setting prices. Decision-making factors involve server capacity, energy consumption, maintenance costs, etc. Adjusting prices affects demand.

### Problem 2: Solution Presentation

Participants are required to prepare a 7-minute presentation to showcase their solution to Problem 1, explaining why the algorithm is strong and reliable in a business context.

## Solution Overview

Our solution employs a **Simulated Annealing (SA)** algorithm to optimize server deployment and pricing strategies. The key components of the solution are:

- **Server Deployment Optimization**: Deciding when and where to purchase, move, or dismiss servers based on capacity, costs, and demand.
- **Pricing Strategy**: Adjusting server prices over time to influence demand and maximize profit.
- **Simulated Annealing**: Utilizing SA to efficiently explore the solution space and escape local optima.

## Repository Structure

- `main.py`: The main script to run the optimization algorithm.
- `real_diff_evaluation.py`: Contains the `DiffSolution` class and related data structures for differential evaluation.
- `real_diff_SA_basic.py`: Implements the Simulated Annealing algorithm and neighborhood operations.
- `data/`: Directory containing input data such as server specifications, datacenter details, and demand forecasts.
- `output/`: Directory where the output JSON files with the optimized server deployment and pricing strategy are saved.
- `environment.yml`: List of Python dependencies required to run the code.

## Getting Started

### Prerequisites

- The code’s feasibility has been verified only on MacBooks with Apple Silicon chips.
- Python 3.8 or higher
- Install the required packages:

```bash
conda env create -f environment.yml
```

### Data Files

Ensure the following data files are present in the `data/` directory:

- `servers.csv`
- `datacenters.csv`
- `demand.csv`
- `price_elasticity_of_demand.csv`
- `selling_prices.csv`

### Running the Code

To execute the optimization algorithm, run:

```bash
python real_diff_SA_PPO.py
```

This will generate an output JSON file in the `output/` directory with the optimized server deployment and pricing strategy.

### Parameters

You can adjust the parameters of the Simulated Annealing algorithm by modifying the following variables in `main.py`:

- `INITIAL_TEMPERATURE`: The starting temperature for SA.
- `MIN_TEMPERATURE`: The minimum temperature to stop the algorithm.
- `ALPHA`: The cooling rate.
- `MAX_ITER`: The maximum number of iterations.
- `GLOBAL_MAX_PURCHASE_RATIO`: The maximum ratio of servers that can be purchased at each step.

### Output

The output JSON file contains two main sections:

- **Fleet**: The server actions (buy, move, dismiss) with time steps and datacenter locations.
- **Pricing Strategy**: The adjusted prices for servers over time.

## Algorithm Details

### Simulated Annealing

Our SA implementation includes several neighborhood operations to explore the solution space:

- **BuyServerOperation**: Purchases new servers.
- **MoveServerOperation**: Moves servers between datacenters.
- **AdjustQuantityOperation**: Adjusts the quantity of servers.
- **AdjustTimeOperation**: Adjusts the timing of server actions.
- **RemoveServerOperation**: Removes servers from the fleet.
- **MergeServersOperation**: Merges servers to optimize resources.
- **AdjustServerPriceOperation**: Adjusts server prices to influence demand.

The algorithm balances exploration and exploitation by probabilistically accepting worse solutions based on the temperature schedule.

### Demand Modeling

The demand is affected by the server prices through a price elasticity model. Adjusting prices allows us to influence demand and optimize profit.

### Cost Components

The total cost includes:

- **Purchase Cost**: Cost of buying servers.
- **Energy Cost**: Cost of energy consumption over time.
- **Maintenance Cost**: Cost of maintaining servers, which increases over time.
- **Moving Cost**: Cost of moving servers between datacenters.

## Results

Our algorithm achieved first place in the algorithm section of the competition, significantly outperforming the second place. However, due to insufficient focus on the PPT and presentation, we ultimately fell short in the finals and finished in third place.


## Acknowledgments

- Huawei Ireland Research Center for organizing the Tech Arena 2024 competition.

---

## Feasible Solution Analysis

List common algorithms and determine their suitability for this problem.

### 1. Optimization and Search Algorithms
- **Simulated Annealing (SA)** — Yes  
- **Genetic Algorithm (GA)** — Yes  
- **Particle Swarm Optimization (PSO)** — Yes  
- **Tabu Search (TS)** — Yes  
- **Ant Colony Optimization (ACO)** — Yes  
- **Gradient Descent** — No  
- **Local Search** — No  
- **Gradient Boosting Trees** — No  
- **Lagrangian Relaxation** — No  
- **Simulated Evolution** — No  
- **Multi-objective Optimization Algorithms (e.g., NSGA-II)** — Yes  
- **Branch and Bound** — No  
- **Integer Linear Programming (ILP)** — Yes  
- **Linear Programming (LP)** — No  
- **Nonlinear Programming (NLP)** — No  
- **Differential Evolution (DE)** — Yes  
- **Bee Algorithm** — No  
- **Cultural Algorithms** — No  
- **Evolution Strategies (ES)** — Yes  
- **Pattern Search** — No  

### 2. Reinforcement Learning Algorithms
- **Proximal Policy Optimization (PPO)** — Yes  
- **Deep Q-Learning (DQN)** — Yes  
- **Double DQN (DDQN)** — No  
- **Trust Region Policy Optimization (TRPO)** — Yes  
- **State-Action-Reward-State-Action (SARSA)** — No  
- **Asynchronous Advantage Actor-Critic (A3C/A2C)** — No  
- **Monte Carlo Tree Search (MCTS)** — Yes  
- **Soft Actor-Critic (SAC)** — Yes  
- **Deep Deterministic Policy Gradient (DDPG)** — No  
- **Actor-Critic (AC)** — No  
- **Hierarchical Reinforcement Learning (HRL)** — No  
- **Distributed DQN** — No  
- **Inverse Reinforcement Learning (IRL)** — No  
- **Multi-Agent Reinforcement Learning (MARL)** — No  

### 3. Other Related Methods
- **Greedy Algorithm** — No  
- **Heuristic Algorithms** — Yes  
- **Evolutionary Programming** — No  
- **Model Predictive Control (MPC)** — Yes  
- **Adaptive Dynamic Programming (ADP)** — No  
- **Priority Queuing Algorithms** — No  
- **Rule-based Systems** — No  
- **Ensemble Learning** — No  
- **Random Forest** — No  
- **Game Theory** — No  

### Highly Feasible Algorithms

Includes SA, PPO, Soft Actor-Critic

#### 1. ✨ Simulated Annealing (SA)
```
Simulated Annealing (SA)
Advantages:
- **Ease of Implementation**: SA is relatively simple to implement by gradually exploring the solution space and probabilistically accepting suboptimal solutions to avoid getting stuck in local optima.
- **Global Optimization Capability**: With enough iterations, SA can find the global optimum, especially suitable for complex problems with many local optima.

Disadvantages:
- **High Computational Overhead**: SA requires repeatedly calling the objective function, leading to significant computational costs on complex problems. Even with optimized evaluation functions, computation time can remain very long.
- **Slow Convergence Speed**: SA’s cooling schedule and the number of iterations directly affect convergence speed. To achieve better solutions, a large number of iterations is often necessary, further increasing computation time.
- **Parameter Sensitivity**: SA’s performance is highly dependent on the selection of parameters like initial temperature and cooling rate. Improper tuning can lead to suboptimal algorithm performance.
```

#### 2. 🌟 Proximal Policy Optimization (PPO) - Reinforcement Learning
```
Advantages:
- **Adaptability to Complex and High-Dimensional State Spaces**: PPO is a reinforcement learning algorithm capable of effective policy learning in high-dimensional state spaces by continuously updating policies through interactions with the environment to learn optimal strategies.
- **Suitable for Dynamic Environments**: PPO can handle environments with uncertainty and dynamic changes, making it suitable for problems requiring long-term planning and multi-step decision-making.
- **Parallel Processing Capability**: PPO can accelerate the learning process by executing policies and environment interactions in parallel through multi-threading or multi-processing.

Disadvantages:
- **Complex Implementation**: Compared to SA, PPO is more complex to implement, requiring the design and tuning of neural network architectures, reward functions, and training processes.
- **Long Training Time**: PPO requires extensive environment interactions and training iterations to converge to a good policy. For highly complex problems, the training time can be very long.
- **Requires Well-Designed Reward Functions**: PPO’s performance heavily relies on the design of the reward function. If the reward function does not effectively reflect the optimization objectives, the learning outcome may be poor.
```

#### 3. Soft Actor-Critic (SAC)
```
**Applicability:**  
SAC is a policy gradient-based reinforcement learning algorithm that maximizes the policy’s entropy to encourage exploration while optimizing long-term rewards. SAC is typically used for complex policy optimization problems in continuous action spaces.  
SAC can handle high-dimensional policy spaces and performs stably in dynamic environments.

**Comparison with SA and PPO:**  
**Advantages:** SAC excels in handling complex policy learning tasks, especially those requiring a balance between exploration and exploitation in long-term planning. Compared to PPO, SAC has stronger exploration capabilities and may find better policies in complex environments.  
**Disadvantages:** SAC’s training process can be more complex than PPO and may require more computational resources and time for tuning.

**Conclusion:** SAC may have stronger exploration capabilities than PPO, especially in dynamic and multi-objective environments when handling complex policy learning tasks. However, SAC’s complexity and training time might prevent it from fully realizing its potential within 14 days. Therefore, PPO might still be a more reliable choice for the current problem.
```

#### 4. Genetic Algorithm (GA)
```
Advantages:
- **Multi-objective Handling**: GA can naturally handle multi-objective optimization problems by considering multiple objectives (e.g., utilization, lifespan, profit) through the fitness function, finding a trade-off solution.
- **Complexity Handling**: GA can manage complex constraints and high-dimensional solution spaces, especially important when dealing with multiple servers and data center operations where GA’s flexibility is crucial.
- **Parallel Acceleration**: You can significantly speed up computation by parallelizing fitness evaluations and population evolution processes. Modern computational resources allow simultaneous evaluation of multiple solutions across multiple processors, enhancing GA’s efficiency.

Disadvantages:
- **Evaluation Function Bottleneck**: Despite parallelization, GA’s performance is still limited by the speed of the evaluation_function. If the evaluation function cannot be optimized to an acceptable speed, GA’s overall efficiency may remain low.
- **Convergence Issues**: In high-dimensional and complex problems, GA may require a large number of generations to converge, meaning that within a limited time, GA might not find solutions very close to the global optimum.
```

### Low Feasibility Algorithms

#### 1. Particle Swarm Optimization (PSO)
```
**Applicability:**  
PSO is a swarm intelligence-based optimization algorithm suitable for both continuous and discrete optimization problems. PSO simulates particles moving through the solution space, using individual and collective best experiences to search for optimal solutions.  
PSO has good global search capabilities for large-scale search spaces but may perform inadequately in complex multi-objective optimization problems, especially when the objective functions are intricate and involve multiple conflicting objectives.

**Comparison with SA and PPO:**  
**Advantages:** PSO is relatively simple to implement and performs well for single-objective or mildly multi-objective problems.  
**Disadvantages:** PSO tends to get trapped in local optima and may require additional enhancements (e.g., dynamic adjustment of inertia weights, velocity updates) to approach the performance of SA and PPO in multi-objective and high-dimensional problems.

**Conclusion:** For the current Huawei problem, PSO may not be as suitable as SA and PPO, particularly considering the complexity of multi-objective optimization.
```

#### 2. Tabu Search (TS)
```
**Applicability:**  
TS is a local search-based optimization algorithm that records recent solutions to prevent the search from getting trapped in local optima. TS performs excellently in discrete optimization problems, especially in solving constrained combinatorial optimization issues.  
TS is suitable for static optimization problems but may have limited performance when dealing with dynamically changing demand curves and multi-objective optimization.

**Comparison with SA and PPO:**  
**Advantages:** TS excels in preventing local optima traps and is suitable for rapidly iterating complex combinatorial optimization problems.  
**Disadvantages:** In multi-objective optimization and long-term planning problems, TS may not handle complex policy learning tasks as effectively as PPO and lacks the global search capabilities of SA.

**Conclusion:** Although TS has advantages in avoiding local optima, SA and PPO have greater potential in multi-objective and dynamic demand environments.
```

#### 3. Ant Colony Optimization (ACO)
```
**Applicability:**  
ACO simulates the foraging behavior of ants by updating pheromones to find the optimal path. ACO performs well in discrete optimization problems, particularly in path planning and resource allocation.  
In multi-objective optimization, ACO can handle different objectives through multiple ant colonies or multiple pheromone update rules. However, due to its iterative process relying on substantial computations, the convergence speed is relatively slow.

**Comparison with SA and PPO:**  
**Advantages:** ACO excels in global search and path selection, suitable for parallel computing and exploring complex solution spaces.  
**Disadvantages:** ACO has a slower convergence speed and may lack the policy learning capabilities of PPO and the flexibility of SA in handling complex multi-objective optimizations.

**Conclusion:** In the current problem, ACO may not be as suitable as SA and PPO, especially considering time constraints and convergence speed.
```

#### 4. Integer Linear Programming (ILP)
```
**Applicability:**  
ILP is an exact optimization method suitable for optimization problems with clear constraints and linear objective functions. By solving linear equations, ILP can find the global optimal solution under constraints.  
In the current problem, if the objective functions and constraints can be linearized, ILP might provide the optimal solution. However, the objective functions are complex and nonlinear, limiting ILP’s applicability.

**Comparison with SA and PPO:**  
**Advantages:** For problems that can be linearized, ILP can provide precise global optimal solutions, especially with clear constraints.  
**Disadvantages:** The current problem involves nonlinear objectives, multiple objectives, and complex cost calculations, making ILP less directly applicable.

**Conclusion:** Due to the problem’s nonlinearity and multi-objective nature, ILP may not be as suitable as SA and PPO in this scenario.
```

#### 5. Differential Evolution (DE)
```
**Applicability:**  
DE is a population-based optimization algorithm primarily used for global optimization problems, particularly performing well in continuous search spaces. DE searches for the global optimum by performing differential mutation and selection within the population.  
DE is suitable for handling high-dimensional, nonlinear, and multi-objective optimization problems but may be less efficient in complex discrete or mixed optimization problems compared to more specialized algorithms.

**Comparison with SA and PPO:**  
**Advantages:** DE performs well in continuous optimization problems and has strong global search capabilities.  
**Disadvantages:** In handling complex discrete problems and multi-objective optimizations, DE’s performance may not match that of SA or PPO. Additionally, its iterative process can be relatively slow.

**Conclusion:** DE may not be as suitable as SA and PPO for the current problem, especially when facing discrete decision spaces and complex multi-objective optimization requirements.
```

#### 6. Trust Region Policy Optimization (TRPO)
```
**Applicability:**  
TRPO is a policy gradient optimization algorithm that maintains the stability of the optimization process by limiting the step size of each policy update. TRPO is particularly suitable for continuous action spaces and policy optimization problems in reinforcement learning.  
TRPO’s advantage lies in balancing exploration and exploitation, making it suitable for complex policy learning and multi-step planning problems.

**Comparison with SA and PPO:**  
**Advantages:** TRPO performs excellently in the reinforcement learning domain, effectively handling instability in policy updates and optimizing within complex policy spaces.  
**Disadvantages:** TRPO has a higher computational overhead and may not perform as well as PPO in discrete action spaces. PPO is considered an improved version of TRPO, typically more efficient and simpler to implement.

**Conclusion:** Although TRPO performs well in policy optimization, PPO, as its improved version, is generally more advantageous in practical applications. Therefore, TRPO may not be as promising as PPO and SA for the current problem.
```

#### 7. Monte Carlo Tree Search (MCTS)
```
**Applicability:**  
MCTS is a tree search and Monte Carlo simulation-based algorithm widely used in decision-making problems and game AI. MCTS builds a decision tree by randomly sampling future possible paths and selects the optimal strategy through iterative exploration.  
MCTS performs well in discrete problems with clear decision paths, especially when considering long-term rewards and uncertainty.

**Comparison with SA and PPO:**  
**Advantages:** MCTS can effectively handle complex problems requiring multi-step decisions and has strong exploration capabilities.  
**Disadvantages:** In environments with limited computational resources and high-dimensional policy spaces, MCTS may face excessive computational overhead. In complex multi-objective optimization problems, MCTS might require numerous simulations to approximate the optimal solution.

**Conclusion:** While MCTS has potential in multi-step decision problems, PPO is more flexible in handling continuous policy learning and multi-objective optimizations. Additionally, SA may be more efficient than MCTS in global searches. Therefore, MCTS may not be as promising as SA and PPO for this problem.
```

#### 8. Evolution Strategies (ES)
```
**Applicability:**  
ES is a population-based optimization algorithm similar to Genetic Algorithms, optimizing solutions through selection, mutation, and recombination operations. ES is typically used for continuous optimization problems but can also adapt to certain discrete problems.  
ES performs well in policy learning and parameter optimization, especially in high-dimensional policy spaces, offering high convergence speed and stability.

**Comparison with SA and PPO:**  
**Advantages:** ES performs well in complex policy optimization tasks, particularly when stable convergence is needed.  
**Disadvantages:** In multi-objective optimization and tasks requiring dynamic adjustments, ES may not be as flexible as PPO and may not perform as well as SA in discrete optimization problems.

**Conclusion:** ES may be similar to PPO in the current problem, but considering PPO’s widespread application and stronger policy learning capabilities, PPO likely has more potential. Therefore, ES may not be as promising as PPO and SA.
```

#### 9. Multi-objective Optimization Algorithms (e.g., NSGA-II)
```
**Applicability:**  
NSGA-II is a genetic algorithm specifically designed for multi-objective optimization, capable of handling multiple conflicting objectives and generating a set of Pareto optimal solutions.  
In the current problem, multiple objectives (e.g., utilization, lifespan, profit) need to be optimized simultaneously. NSGA-II can effectively manage these competing objectives and generate various balanced solutions for selection.

**Comparison with SA and PPO:**  
**Advantages:** NSGA-II has significant advantages in handling multi-objective optimization problems, generating Pareto front solutions for each objective. This is highly useful when multiple objectives need to be considered comprehensively.  
**Disadvantages:** NSGA-II may require substantial computational resources and time to converge. Additionally, in dynamic demand scenarios, frequent adjustments and re-evaluations may be necessary.
```

## Why Does the Evaluation Function Become a Bottleneck?

Yes, regardless of the algorithm or optimization strategy used, if the evaluation function is computed very slowly, it becomes the bottleneck of the overall solution. This is because nearly all optimization algorithms require frequent calls to the evaluation function to assess the quality of current solutions and proceed with iterations or searches. The performance of the evaluation function directly impacts the overall efficiency and feasibility of the algorithm.

### 1. **Frequent Calls**
Whether it’s Simulated Annealing (SA), Genetic Algorithms (GA), Reinforcement Learning (PPO), or other optimization algorithms, they all rely on frequent evaluations of candidate solutions. Each iteration or state update involves calling the evaluation function to compute the objective function value. If the evaluation function is slow, the computation time for each iteration becomes lengthy, thereby extending the entire optimization process.

### 2. **Scale Effect**
The slowness of the evaluation function not only affects single evaluation times but also accumulates over numerous iterations. For example:
- In Genetic Algorithms, the evaluation function needs to assess every individual in each generation’s population, meaning that the slowness of the evaluation function can exponentially amplify with population size and the number of generations.
- In Simulated Annealing, the slowness directly impacts the entire annealing process’s efficiency, leading to longer convergence times.
- In Reinforcement Learning, policy updates depend on a large amount of interaction data, and slow evaluation functions result in inefficient policy learning.

## Strategies to Mitigate Slow Evaluation Functions

### 1. **Optimize the Evaluation Function**
- **Vectorized Computations**: Convert row-by-row computations into vectorized operations whenever possible, utilizing efficient array operations to accelerate calculations.
- **Cache Redundant Calculations**: Cache or precompute parts of the evaluation function that do not need to be recalculated every time, reducing unnecessary computations.
- **Parallelize Computations**: Parallelize independent computation steps using multi-threading or multi-processing to speed up calculations.
- **Simplify Logic**: Analyze the evaluation function’s logic to remove redundant or unnecessary parts, simplifying the computational process.

### 2. **Use Surrogate Models**
- **Surrogate Models**: Employ a fast-computing surrogate model to replace the original evaluation function. For instance, machine learning models (like regression models or neural networks) can predict evaluation function values, significantly increasing computation speed at the expense of some accuracy.

### 3. **Reduce Evaluation Frequency**
- **Sparse Evaluation**: Skip calling the evaluation function in some iterations or only evaluate a subset of individuals, thereby reducing the number of evaluation function calls.
- **Heuristic Pre-filtering**: Use simple heuristic rules to filter out candidate solutions that are clearly not optimal before performing detailed evaluations, thereby reducing the number of solutions that need to be thoroughly evaluated.
