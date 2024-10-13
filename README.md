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

## 可行方案分析

常见算法列举，并判断是否合适本题目？

1. 优化与搜索算法
模拟退火（Simulated Annealing, SA）——是  
遗传算法（Genetic Algorithm, GA）——是  
粒子群优化（Particle Swarm Optimization, PSO）——是  
禁忌搜索（Tabu Search, TS）——是  
蚁群优化（Ant Colony Optimization, ACO）——是  
梯度下降（Gradient Descent）——否  
局部搜索（Local Search）——否  
梯度增强树（Gradient Boosting Trees）——否  
拉格朗日松弛法（Lagrangian Relaxation）——否  
模拟进化（Simulated Evolution）——否  
多目标优化算法（Multi-objective Optimization Algorithms, 如NSGA-II）——是  
分支定界法（Branch and Bound）——否  
整数线性规划（Integer Linear Programming, ILP）——是  
线性规划（Linear Programming, LP）——否  
非线性规划（Nonlinear Programming, NLP）——否  
差分进化（Differential Evolution, DE）——是
蜂群优化（Bee Algorithm）——否  
文化算法（Cultural Algorithms）——否  
进化策略（Evolution Strategies, ES）——是  
模式搜索（Pattern Search）——否  

2. 强化学习算法
近端策略优化（Proximal Policy Optimization, PPO）——是  
深度Q学习（Deep Q-Learning, DQN）——是  
双重深度Q学习（Double DQN, DDQN）——否  
信赖域策略优化（Trust Region Policy Optimization, TRPO）——是  
状态动作回报期望算法（State-Action-Reward-State-Action, SARSA）——否  
倾斜概率策略梯度法（Asynchronous Advantage Actor-Critic, A3C/A2C）——否  
蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）——是  
软演员评论家算法（Soft Actor-Critic, SAC）——是  
深度确定性策略梯度（Deep Deterministic Policy Gradient, DDPG）——否  
演员-评论家算法（Actor-Critic, AC）——否  
分层强化学习（Hierarchical Reinforcement Learning, HRL）——否  
分布式深度Q学习（Distributed DQN）——否  
逆强化学习（Inverse Reinforcement Learning, IRL）——否  
多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）——否  

3. 其他相关方法
贪心算法（Greedy Algorithm）——否  
启发式算法（Heuristic Algorithms）——是  
进化规划（Evolutionary Programming）——否  
模型预测控制（Model Predictive Control, MPC）——是  
自适应动态规划（Adaptive Dynamic Programming, ADP）——否  
优先级排队算法（Priority Queuing Algorithms）——否  
基于规则的系统（Rule-based Systems）——否  
集成学习（Ensemble Learning）——否  
随机森林（Random Forest）——否  
博弈论（Game Theory）——否  

### 高可行性算法

包括SA， PPO， Soft Actor-Critic

1. ✨SA 模拟退火
```
模拟退火（SA）
优点：
简单易实现：SA的实现相对简单，通过逐步探索解空间并基于概率接受次优解，避免陷入局部最优。
全局优化能力：SA在足够多的迭代下有能力找到全局最优解，尤其适用于有大量局部最优解的复杂问题。
缺点：
计算开销高：SA 需要反复调用目标函数，在复杂问题上计算开销非常大。即使优化了评估函数，计算时间仍可能非常长。
收敛速度慢：SA 的降温策略和迭代次数直接影响收敛速度，但为了获得更好的解，往往需要大量的迭代，这进一步增加了计算时间。
参数敏感性：SA 的性能高度依赖于初始温度、降温系数等参数的选择，调整不当可能导致算法效果不佳。
```

2. 🌟PPO 强化学习
```
优点：
适应复杂和高维状态空间：PPO 是一种强化学习算法，能够在高维状态空间中进行有效的策略学习，通过不断的策略更新和环境交互来学习最优策略。
适合动态环境：PPO 能够处理带有不确定性和动态变化的环境，适用于需要长期规划和多步骤决策的问题。
并行处理能力：PPO 可以通过多线程或多进程并行执行策略和环境交互，加速学习过程。
缺点：
实现复杂：相比于 SA，PPO 的实现更为复杂，需要设计和调试神经网络结构、奖励函数、训练过程等。
训练时间长：PPO 需要大量的环境交互和训练迭代才能收敛到一个好的策略，对于高复杂度问题，训练时间可能会非常长。
需要良好的奖励函数设计：PPO 的表现高度依赖于奖励函数的设计，如果奖励函数无法有效反映目标优化需求，可能导致学习效果不佳。
```

3. 软演员评论家算法（Soft Actor-Critic, SAC）  
```
适用性：  
SAC 是一种基于策略梯度的强化学习算法，通过最大化策略的熵来鼓励策略的探索，同时优化长期回报。SAC 通常用于连续动作空间中的复杂策略优化问题。  
SAC 能够处理高维度的策略空间，并且在动态变化的环境中表现稳定。  
对比SA和PPO：  
优势：SAC在处理复杂策略学习任务中具有优异表现，特别是在需要平衡探索和利用的长期规划任务中。相比PPO，SAC的探索性更强，可能在复杂环境下找到更优的策略。  
劣势：SAC的训练过程可能比PPO更复杂，并且需要更多的计算资源和时间来调优。  
结论：SAC在处理复杂策略学习任务时可能比PPO具有更强的探索能力，尤其是在动态和多目标环境中。然而，SAC的复杂性和训练时间可能使其在14天内难以充分发挥潜力。因此，  PPO在当前题目中可能仍然是更稳妥的选择。 
``` 

4. GA
```
优点
多目标处理：GA 可以自然地处理多目标优化问题，通过适应度函数综合考虑多个目标（如利用率、寿命、利润）的权重，找到一个折中的解。

复杂性处理：GA 能够处理复杂的约束和高维度的解空间，尤其是在涉及多种服务器和数据中心操作时，GA 的灵活性表现得尤为重要。

并行加速：你可以通过并行化适应度评估和种群进化过程，显著加速计算。现代计算资源允许在多个处理器上同时评估多个解，从而提升 GA 的效率。

缺点
评估函数的瓶颈：尽管 GA 可以并行化，但其性能仍然受到 evaluation_function 计算速度的限制。如果 evaluation_function 无法优化到一个可接受的速度，GA 的整体效率仍然可能较低。

收敛性问题：在高维复杂问题上，GA 可能需要大量的代数才能收敛，这意味着在有限的时间内，GA 可能无法找到非常接近全局最优的解。
```

### 低可行性算法

1. 粒子群优化（Particle Swarm Optimization, PSO）  
适用性：  
PSO 是一种基于群体智能的优化算法，适用于连续和离散优化问题。PSO通过模拟粒子在解空间中的移动，利用个体和群体的最优经验进行搜索。  
PSO在处理大规模搜索空间时，具有较好的全局搜索能力，但在复杂多目标优化问题中可能表现不足，尤其是在目标函数复杂、涉及多个相互冲突的目标时。  
对比SA和PPO：  
优势：PSO的实现相对简单，且在处理单目标或轻度多目标问题时表现良好。  
劣势：PSO容易陷入局部最优，且在多目标优化和高维度问题上，可能需要额外的改进（如动态调整惯性权重、速度更新等）才能接近SA和PPO的表现。  
结论：在当前华为题目上，PSO可能不如SA和PPO适合，尤其是考虑到多目标优化的复杂性。  

2. 禁忌搜索（Tabu Search, TS）  
适用性：  
TS 是一种基于局部搜索的优化算法，通过记录最近的解，防止搜索陷入局部最优。TS可以在离散优化问题中表现出色，尤其是在解决具有约束的组合优化问题时。  
TS适合处理静态的优化问题，但面对动态变化的需求曲线和多目标优化时，TS的性能可能受到限制。  
对比SA和PPO：  
优势：TS在防止局部最优方面表现优异，适合快速迭代求解复杂的组合优化问题。  
劣势：在多目标优化和长期规划问题中，TS可能无法像PPO那样处理复杂的策略学习任务，且相比SA的全局搜索能力，TS的搜索空间更有限。  
结论：虽然TS在防止局部最优方面有优势，但在多目标和动态需求环境下，SA和PPO更具潜力。  

3. 蚁群优化（Ant Colony Optimization, ACO）  
适用性：  
ACO 模拟蚂蚁觅食的行为，通过信息素的更新来寻找最优路径。ACO在离散优化问题中，尤其是路径规划和资源分配问题中表现良好。  
在多目标优化中，ACO可以通过多个蚁群或信息素的多重更新规则处理不同的目标，但由于其迭代过程依赖于大量的计算，收敛速度较慢。  
对比SA和PPO：  
优势：ACO在全局搜索和路径选择方面表现出色，适合并行计算，能够探索复杂解空间。  
劣势：ACO的收敛速度较慢，且在处理复杂多目标优化时，可能不如PPO具备的策略学习能力和SA的灵活性。  
结论：在当前题目中，ACO可能不如SA和PPO适合，尤其是在考虑到时间限制和收敛速度的情况下。  

4. 整数线性规划（Integer Linear Programming, ILP）  
适用性：  
ILP 是一种精确优化方法，适用于具有明确约束条件和线性目标函数的优化问题。通过求解线性方程组，可以在约束条件下找到全局最优解。  
在当前题目中，如果目标函数和约束条件可以线性化，ILP可能提供最优解。然而，当前题目的目标函数复杂且非线性，ILP的适用性受到限制。  
对比SA和PPO：  
优势：在适合线性化的问题中，ILP可以提供精确的全局最优解，特别是在有明确约束的情况下。  
劣势：当前题目的目标函数是非线性的，涉及多目标和复杂的成本计算，ILP可能无法直接应用。  
结论：由于题目的非线性和多目标特性，ILP在这个场景中可能不如SA和PPO适合。  

5. 差分进化（Differential Evolution, DE）  
适用性：  
DE 是一种基于种群的优化算法，主要用于全局优化问题，特别是在连续搜索空间中表现良好。DE通过对种群内的解进行差分变异和选择来寻找全局最优解。  
DE 适合处理高维度、非线性和多目标优化问题，但其效率在处理复杂的离散或混合优化问题时可能不如一些更专门的算法。  
对比SA和PPO：  
优势：DE在处理连续优化问题时表现良好，并且具有强大的全局搜索能力。  
劣势：在处理复杂的离散问题和多目标优化问题时，DE的性能可能不如SA或PPO，而且其迭代过程可能比较缓慢。  
结论：DE在当前题目中可能不如SA和PPO适合，尤其是面对离散的决策空间和复杂的多目标优化需求时。  

7. 信赖域策略优化（Trust Region Policy Optimization, TRPO）  
适用性：  
TRPO 是一种策略梯度优化算法，通过限制每次策略更新的步幅，保持优化过程的稳定性，避免策略更新过度。TRPO 尤其适用于强化学习中的连续动作空间和策略优化问题。  
TRPO 的优点在于能够平衡探索和利用，适用于复杂的策略学习和多步长规划问题。  
对比SA和PPO：  
优势：TRPO在强化学习领域表现优秀，能够有效处理策略更新中的不稳定性，并在复杂的策略空间中进行优化。  
劣势：TRPO 的计算开销较大，且在离散动作空间中的表现可能不如PPO优异。PPO被视为TRPO的改进版本，通常更为高效且实现更为简单。  
结论：虽然TRPO在策略优化中表现良好，但PPO作为其改进版，通常在实际应用中更为优越。因此，TRPO在当前题目中可能不如PPO和SA更有希望。  

8. 蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）  
适用性：  
MCTS 是一种基于树搜索和蒙特卡洛模拟的算法，广泛用于决策问题和游戏AI中。MCTS通过随机采样未来的可能路径，逐步构建决策树，并选择最优策略。  
MCTS在处理具有明确决策路径的离散问题时表现良好，特别是当需要考虑长远收益和不确定性时。  
对比SA和PPO：  
优势：MCTS能够有效处理需要多步决策的复杂问题，并且具有强大的探索能力。  
劣势：MCTS在计算资源有限和高维度策略空间中，可能面临计算开销过大的问题。在复杂多目标优化问题中，MCTS可能需要大量的模拟次数来逼近最优解。  
结论：MCTS在多步决策问题中具有潜力，但在当前问题中，PPO在处理连续策略学习和多目标优化时可能更为灵活。SA在处理全局搜索时的效率也可能优于MCTS。因此，MCTS在这个题目中可能不如SA和PPO更有希望。  

9. 进化策略（Evolution Strategies, ES）  
适用性：  
ES 是一种基于种群的优化算法，类似于遗传算法，通过选择、变异和重组操作来优化解。ES 通常用于解决连续优化问题，但也可以适应某些离散问题。  
ES 在策略学习和参数优化方面表现良好，尤其是在高维度的策略空间中，其收敛速度和稳定性较高。  
对比SA和PPO：  
优势：ES在处理复杂策略优化问题时表现良好，特别是在需要稳定收敛的情况下。  
劣势：在面对多目标优化和需要动态调整的任务时，ES可能不如PPO那样灵活，也可能不如SA在离散优化问题中的表现。  
结论：ES在当前题目中可能与PPO相似，但考虑到PPO的广泛应用和更强的策略学习能力，PPO可能更有潜力。因此，ES可能不如PPO和SA更有希望。  

4. 多目标优化算法（如 NSGA-II ）  
适用性：  
NSGA-II 是一种专门用于多目标优化的遗传算法，能够处理多个相互冲突的目标，并生成一组Pareto最优解。  
在当前题目中，多个目标（如利用率、寿命、利润）需要同时优化，NSGA-II能够很好地处理这些相互竞争的目标，并生成多种平衡解供选择。  
对比SA和PPO：  
优势：NSGA-II在处理多目标优化问题上有显著优势，可以为每个目标生成Pareto前沿解，这在需要综合考虑多种目标时非常有用。  
劣势：NSGA-II可能需要大量的计算资源和时间来收敛，且在处理动态需求时，可能需要频繁调整和重新评估。  


# 为什么评估函数成为瓶颈？

是的，无论使用哪种算法或优化策略，如果评估函数计算得非常慢，它都会成为整体方案的瓶颈。这是因为几乎所有的优化算法都需要频繁调用评估函数来判断当前解的质量，并据此进行迭代或搜索。评估函数的性能直接影响算法的整体效率和可行性。

#### 1. **频繁调用**
无论是模拟退火（SA）、遗传算法（GA）、强化学习（PPO），还是其他优化算法，它们都依赖于对候选解的频繁评估。每次迭代或每次状态更新时，评估函数都会被调用来计算目标函数值。如果评估函数很慢，那么每次迭代的计算时间就会很长，从而延长整个优化过程。

#### 2. **规模效应**
评估函数的慢速不仅影响单次评估时间，还会在大量的迭代中累计放大。例如：
- 在遗传算法中，评估函数需要对每一代种群中的所有个体进行评估，这意味着评估函数的慢速会在种群规模和迭代次数上成倍放大。
- 在模拟退火中，评估函数的慢速会直接影响整个退火过程的效率，导致收敛时间变长。
- 在强化学习中，策略更新依赖于大量的交互数据，评估函数的慢速会导致策略学习效率低下。

### 解决评估函数慢速的策略

#### 1. **优化评估函数**
- **矢量化计算**：尽可能将逐行计算转换为矢量化操作，利用高效的数组操作来加速计算。
- **缓存重复计算**：对于不需要每次都重新计算的部分，可以进行缓存或预计算，以减少不必要的计算。
- **并行化计算**：将独立的计算步骤并行化，利用多线程或多进程加速计算。
- **简化逻辑**：分析评估函数的逻辑，去除冗余或不必要的部分，以简化计算过程。

#### 2. **使用代理模型**
- **代理模型（Surrogate Models）**：使用一个快速计算的代理模型来代替原始的评估函数。例如，可以使用机器学习模型（如回归模型或神经网络）来预测评估函数值，尽管精度可能稍有降低，但计算速度可以大幅提高。

#### 3. **减少评估频率**
- **稀疏评估**：在某些迭代中跳过评估函数的调用，或者只对部分个体进行评估，从而减少评估函数的调用次数。
- **基于启发式的初筛**：在进行评估之前，通过一些简单的启发式规则过滤掉显然不可能是优解的候选解，以减少需要详细评估的解的数量。

### 最终结论

**是的，评估函数的计算速度对所有方案来说都是至关重要的。** 无论使用哪种优化算法，评估函数的慢速都会成为瓶颈，显著影响整体计算效率。因此，在任何优化方案中，**加速评估函数的计算**是最优先的优化任务。如果无法显著加快评估函数的计算速度，可能需要重新考虑算法选择，或采用一些替代方法如代理模型来解决问题。
