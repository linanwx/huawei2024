# server_management_env.py

import gym
from gym import spaces
import numpy as np
import copy
import random
import pandas as pd
from typing import Dict
from real_diff_evaluation import (
    DiffSolution,
    ServerInfo,
    ServerMoveInfo,
    TIME_STEPS,
    SERVER_GENERATION_MAP,
    LATENCY_SENSITIVITY_MAP,
    datacenter_info_dict,
    selling_price_dict,
)

# Load necessary data
datacenter_ids = list(datacenter_info_dict.keys())

class ServerManagementEnv(gym.Env):
    """
    Custom Environment for the server management system.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, seed=None):
        super(ServerManagementEnv, self).__init__()

        # Initialize the DiffSolution instance
        self.seed_value = seed if seed is not None else random.randint(0, 10000)
        self.seed(self.seed_value)
        self.solution = DiffSolution(seed=self.seed_value)
        self.slot_manager = self.solution.slot_manager  # Assuming slot_manager is initialized in DiffSolution

        # Define action space components
        self.server_types = list(SERVER_GENERATION_MAP.keys())
        self.num_server_types = len(self.server_types)
        self.max_quantity = 5  # Maximum quantity to buy at once
        self.datacenter_ids = datacenter_ids
        self.num_datacenters = len(self.datacenter_ids)

        # Multi-discrete action space: [server_type, quantity, datacenter]
        self.action_space = spaces.MultiDiscrete([self.num_server_types, self.max_quantity, self.num_datacenters])

        # Observation space: Flattened capacity_matrix, average_utilization, datacenter_slots, and demand
        capacity_shape = self.solution._DiffSolution__capacity_matrix.shape
        capacity_size = np.prod(capacity_shape)

        utilization_size = self.solution._DiffSolution__average_utilization.size

        datacenter_slots_shape = self.slot_manager.datacenter_slots.shape
        datacenter_slots_size = np.prod(datacenter_slots_shape)

        # For demand, assuming a vector per time step
        demand_size = len(LATENCY_SENSITIVITY_MAP) * len(SERVER_GENERATION_MAP)

        # Total observation size
        obs_size = capacity_size + utilization_size + datacenter_slots_size + demand_size

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )

        self.state = None
        self.current_time_step = 0
        self.done = False

        # Load demand data
        self.demand_data = self.load_demand_data()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)

    def load_demand_data(self):
        # Implement the method to load actual demand data
        # For example, from a CSV file or as a generated dataset
        # Here, we'll create a dummy demand data array
        demand_data = np.zeros((TIME_STEPS, len(LATENCY_SENSITIVITY_MAP), len(SERVER_GENERATION_MAP)))
        # Populate demand_data with actual values as needed
        return demand_data

    def get_current_demand(self):
        # Return the demand at the current time step
        if self.current_time_step < TIME_STEPS:
            return self.demand_data[self.current_time_step].flatten()
        else:
            return np.zeros(len(LATENCY_SENSITIVITY_MAP) * len(SERVER_GENERATION_MAP))

    def reset(self):
        # Reset the environment to an initial state
        self.solution = DiffSolution(seed=self.seed_value)
        self.slot_manager = self.solution.slot_manager
        self.current_time_step = 0
        self.done = False

        # Get the initial observation
        self.state = self._get_observation()

        return self.state

    def _get_observation(self):
        # Gather and flatten the necessary state variables
        capacity_matrix = self.solution._DiffSolution__capacity_matrix.flatten()
        average_utilization = self.solution._DiffSolution__average_utilization

        datacenter_slots = self.slot_manager.datacenter_slots.flatten()

        demand = self.get_current_demand()

        observation = np.concatenate([
            capacity_matrix,
            average_utilization,
            datacenter_slots,
            demand,
        ])

        return observation.astype(np.float32)

    def step(self, action):
        """
        Execute one time step within the environment
        """
        # Reset any previous changes
        self.solution.discard_server_changes()

        # Apply the action
        server_idx, quantity, datacenter_idx = action
        server_generation = self.server_types[server_idx]
        datacenter_id = self.datacenter_ids[datacenter_idx]

        # Check if slots are available
        slots_needed = quantity  # Assuming one slot per server
        available_slots = self.slot_manager.get_available_slots(datacenter_id)
        if available_slots >= slots_needed:
            # "Buy" action: Buy the specified quantity of servers of the selected type
            for _ in range(quantity + 1):  # quantity ranges from 0 to max_quantity - 1
                server_id = f"server_{np.random.randint(1000,9999)}"
                buy_move_info = [ServerMoveInfo(time_step=self.current_time_step, target_datacenter=datacenter_id)]

                server_info = ServerInfo(
                    server_id=server_id,
                    server_generation=server_generation,
                    quantity=1,
                    dismiss_time=min(self.current_time_step + 50, TIME_STEPS - 1),
                    buy_and_move_info=buy_move_info
                )

                # Apply the "buy" operation
                self.solution.apply_server_change(server_info)
            self.solution.commit_server_changes()
        else:
            # Cannot buy due to slot limitations
            pass  # Action has no effect or apply a penalty in the reward

        # Update the state
        self.current_time_step += 1

        # Compute the reward
        reward = self._compute_reward()

        # Get new observation
        self.state = self._get_observation()

        # Check if the episode is done
        if self.current_time_step >= TIME_STEPS:
            self.done = True

        info = {}

        return self.state, reward, self.done, info

    def calculate_immediate_cost(self):
        # Calculate costs directly associated with the current action
        # For simplicity, let's consider the purchase price of the servers bought
        immediate_cost = 0.0
        for server_id, server_info in self.solution.server_map.items():
            if server_info.buy_and_move_info[0].time_step == self.current_time_step - 1:
                immediate_cost += server_info.purchase_price * server_info.quantity
        return immediate_cost

    def calculate_immediate_revenue(self):
        # Estimate revenue generated in the current time step
        # Assuming revenue is generated based on capacity and demand fulfillment
        # For simplicity, let's return a placeholder value
        immediate_revenue = 0.0
        # Calculate revenue based on capacity used to meet demand
        # Implement the actual logic based on your system
        return immediate_revenue

    def calculate_unmet_demand_penalty(self):
        # Calculate penalties for unmet demand
        unmet_demand_penalty = 0.0
        # Compare capacity with demand and calculate penalty
        # Implement the actual logic based on your system
        return unmet_demand_penalty

    def _compute_reward(self):
        # Compute the reward based on the immediate cost, revenue, and penalties
        immediate_cost = self.calculate_immediate_cost()
        immediate_revenue = self.calculate_immediate_revenue()
        unmet_demand_penalty = self.calculate_unmet_demand_penalty()

        reward = immediate_revenue - immediate_cost - unmet_demand_penalty

        return reward

    def render(self, mode="human"):
        pass  # Implement render logic if needed

    def close(self):
        pass