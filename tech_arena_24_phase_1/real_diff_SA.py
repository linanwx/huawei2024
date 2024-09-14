import os
import time
import copy
import random
import threading
import numpy as np
import pandas as pd
from datetime import datetime

# Import the new DiffSolution and related classes
from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo  # Assume code snippet 1 is in this module

TIME_STEPS = 168
MAX_BUY_PERCENTAGE = 0.12
MAX_ADD_PERCENTAGE = 0.12

# Automatically create output directory
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate a timestamped log filename
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'{output_dir}simulation_{current_time}.log'

class ThreadSafeIDGenerator:
    def __init__(self, start=0):
        self.current_id = start
        self.lock = threading.Lock()

    def next_id(self):
        with self.lock:
            self.current_id += 1
            return str(self.current_id)

class SlotAvailabilityManager:
    def __init__(self, datacenters, verbose=False):
        self.verbose = verbose
        self.datacenter_slots = {}
        for _, row in datacenters.iterrows():
            dc_id = row['datacenter_id']
            slots_capacity = row['slots_capacity']
            # Initialize a numpy array for each datacenter to represent slots over time
            self.datacenter_slots[dc_id] = np.full(TIME_STEPS, slots_capacity, dtype=int)
        self.total_slots = {dc: datacenters.loc[datacenters['datacenter_id'] == dc, 'slots_capacity'].values[0] for dc in datacenters['datacenter_id']}

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    def check_availability(self, start_time, end_time, data_center, slots_needed):
        slots = self.datacenter_slots[data_center][start_time:end_time+1]
        return np.all(slots >= slots_needed)

    def get_maximum_available_slots(self, start_time, end_time, data_center):
        slots = self.datacenter_slots[data_center][start_time:end_time+1]
        return np.min(slots)

    def update_slots(self, start_time, end_time, data_center, slots_needed, operation='buy'):
        if operation == 'buy':
            self.datacenter_slots[data_center][start_time:end_time+1] -= slots_needed
        elif operation == 'cancel':
            self.datacenter_slots[data_center][start_time:end_time+1] += slots_needed
        self._print(f"Updated slots in datacenter {data_center} from time {start_time} to {end_time} with operation '{operation}' and slots_needed {slots_needed}")

    def check_slot_consistency(self, solution_servers):
        # Reconstruct slot usage from the solution
        reconstructed_slots = {dc_id: np.zeros(TIME_STEPS, dtype=int) for dc_id in self.datacenter_slots.keys()}
        for server in solution_servers.values():
            quantity = server.quantity
            slots_size = server.slots_size
            total_slots = quantity * slots_size
            move_info = server.move_info
            dismiss_time = server.dismiss_time
            for i in range(len(move_info)):
                move_time = move_info[i].time_step
                data_center = move_info[i].target_datacenter
                if i + 1 < len(move_info):
                    end_time = move_info[i + 1].time_step - 1
                else:
                    end_time = dismiss_time
                reconstructed_slots[data_center][move_time:end_time+1] += total_slots

        # Compare reconstructed slots with current slots
        for dc_id, slots in self.datacenter_slots.items():
            expected_slots = self.total_slots[dc_id] - slots
            if not np.array_equal(expected_slots, reconstructed_slots[dc_id]):
                self._print(f"Slot inconsistency detected in datacenter {dc_id}")
                return False
        return True

class SimulatedAnnealing:
    def __init__(self, slot_manager, seed, initial_temp, min_temp, alpha, max_iter, verbose=False):
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.id_gen = ThreadSafeIDGenerator(start=0)
        self.slot_manager:SlotAvailabilityManager = slot_manager
        self.verbose = verbose
        self.seed = seed
        self.S = DiffSolution(seed=seed, verbose=verbose)
        random.seed(seed)
        np.random.seed(seed)

        # Fixed operation probabilities
        self.operations = ['buy', 'adjust_dismiss_time', 'adjust_quantity', 'move']
        self.operation_probabilities = [0.4, 0.2, 0.2, 0.2]

        self.best_solution = copy.deepcopy(self.S)
        self.best_score = float('-inf')

        # Load servers and datacenters data
        self.servers_df = pd.read_csv('./data/servers.csv')
        self.datacenters_df = pd.read_csv('./data/datacenters.csv')

        self.pending_slot_updates = []

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def choose_operation(self):
        return random.choices(self.operations, weights=self.operation_probabilities, k=1)[0]

    def generate_neighbor(self):
        self.pending_slot_updates = []
        operation = self.choose_operation()
        if operation == 'buy':
            success = self.buy_server()
        elif operation == 'adjust_purchase_time':
            success = self.adjust_purchase_time()
        elif operation == 'adjust_dismiss_time':
            success = self.adjust_dismiss_time()
        elif operation == 'adjust_quantity':
            success = self.adjust_quantity()
        elif operation == 'move':
            success = self.move_server()
        else:
            success = False
        return success

    def buy_server(self):
        time_step = random.randint(0, TIME_STEPS - 1)
        data_center = random.choice(list(self.slot_manager.total_slots.keys()))

        # Get available servers at the time_step
        available_servers = self.servers_df[
            (self.servers_df['release_time'].apply(lambda x: eval(x)[0]) <= time_step + 1) &
            (self.servers_df['release_time'].apply(lambda x: eval(x)[1]) >= time_step + 1)
        ]

        if available_servers.empty:
            self._print("No available servers at time step", time_step)
            return False

        selected_server = available_servers.sample(n=1).iloc[0]
        server_generation = selected_server['server_generation']
        life_expectancy = selected_server['life_expectancy']
        slots_size = selected_server['slots_size']

        # Decide on dismiss life
        if random.random() < 0.5:
            dismiss_life = random.randint(1, life_expectancy)
        else:
            dismiss_life = life_expectancy

        dismiss_time = min(time_step + dismiss_life - 1, TIME_STEPS - 1)

        # Calculate required slots
        total_slots_needed = slots_size

        # Check slot availability
        if self.slot_manager.check_availability(time_step, dismiss_time, data_center, total_slots_needed):
            # Create new ServerInfo and apply
            server_id = self.id_gen.next_id()
            move_info = [ServerMoveInfo(time_step=time_step, target_datacenter=data_center)]
            server_info = ServerInfo(
                server_id=server_id,
                server_generation=server_generation,
                quantity=1,
                dismiss_time=dismiss_time,
                move_info=move_info
            )
            try:
                self.S.apply_server_change(server_info)
            except ValueError as e:
                self._print(f"Error applying server change: {e}")
                return False

            # Collect slot updates
            self.pending_slot_updates.append(('buy', time_step, dismiss_time, data_center, total_slots_needed))

            self._print(f"Bought server {server_id} at time {time_step} in datacenter {data_center}")
            return True
        else:
            self._print(f"Not enough slots in datacenter {data_center} for server {server_generation}")
            return False

    def adjust_dismiss_time(self):
        if not self.S.server_map:
            self._print("No servers to adjust dismiss time")
            return False

        server = random.choice(list(self.S.server_map.values()))
        server_id = server.server_id
        current_dismiss_time = server.dismiss_time
        life_expectancy = server.life_expectancy
        time_step = server.move_info[0].time_step
        data_center = server.move_info[-1].target_datacenter
        slots_size = server.slots_size
        quantity = server.quantity

        # Decide to increase or decrease
        if random.random() < 0.5:
            # Decrease dismiss time
            if current_dismiss_time <= time_step:
                self._print("Cannot decrease dismiss time further")
                return False
            new_dismiss_time = random.randint(time_step, current_dismiss_time - 1)
            # Collect slot updates
            total_slots_released = (current_dismiss_time - new_dismiss_time) * slots_size * quantity
            self.pending_slot_updates.append(('cancel', new_dismiss_time + 1, current_dismiss_time, data_center, slots_size * quantity))
            # Update server
            server_copy = copy.deepcopy(server)
            server_copy.dismiss_time = new_dismiss_time
            try:
                self.S.apply_server_change(server_copy)
            except ValueError as e:
                self._print(f"Error applying server change: {e}")
                return False
            self._print(f"Decreased dismiss time of server {server_id} to {new_dismiss_time}")
            return True
        else:
            # Increase dismiss time
            max_dismiss_time = min(time_step + life_expectancy - 1, TIME_STEPS - 1)
            if current_dismiss_time >= max_dismiss_time:
                self._print("Cannot increase dismiss time further")
                return False
            new_dismiss_time = random.randint(current_dismiss_time + 1, max_dismiss_time)
            # Check slot availability
            if self.slot_manager.check_availability(current_dismiss_time + 1, new_dismiss_time, data_center, slots_size * quantity):
                # Collect slot updates
                total_slots_needed = (new_dismiss_time - current_dismiss_time) * slots_size * quantity
                self.pending_slot_updates.append(('buy', current_dismiss_time + 1, new_dismiss_time, data_center, slots_size * quantity))
                # Update server
                server_copy = copy.deepcopy(server)
                server_copy.dismiss_time = new_dismiss_time
                try:
                    self.S.apply_server_change(server_copy)
                except ValueError as e:
                    self._print(f"Error applying server change: {e}")
                    return False
                self._print(f"Increased dismiss time of server {server_id} to {new_dismiss_time}")
                return True
            else:
                self._print("Not enough slots to extend dismiss time")
                return False

    def adjust_quantity(self):
        if not self.S.server_map:
            self._print("No servers to adjust quantity")
            return False

        server = random.choice(list(self.S.server_map.values()))
        server_id = server.server_id
        current_quantity = server.quantity
        time_step = server.move_info[0].time_step
        dismiss_time = server.dismiss_time
        data_center = server.move_info[-1].target_datacenter
        slots_size = server.slots_size

        # Decide to increase or decrease
        if random.random() < 0.5:
            # Decrease quantity
            if current_quantity <= 1:
                self._print("Cannot decrease quantity further")
                return False
            new_quantity = random.randint(1, current_quantity - 1)
            quantity_change = current_quantity - new_quantity
            # Collect slot updates
            total_slots_released = slots_size * quantity_change
            self.pending_slot_updates.append(('cancel', time_step, dismiss_time, data_center, total_slots_released))
            # Update server
            server_copy = copy.deepcopy(server)
            server_copy.quantity = new_quantity
            try:
                self.S.apply_server_change(server_copy)
            except ValueError as e:
                self._print(f"Error applying server change: {e}")
                return False
            self._print(f"Decreased quantity of server {server_id} to {new_quantity}")
            return True
        else:
            # Increase quantity
            max_additional_quantity = int(self.slot_manager.get_maximum_available_slots(time_step, dismiss_time, data_center) / slots_size)
            if max_additional_quantity <= 0:
                self._print("Not enough slots to increase quantity")
                return False
            additional_quantity = random.randint(1, max_additional_quantity)
            new_quantity = current_quantity + additional_quantity
            # Collect slot updates
            total_slots_needed = slots_size * additional_quantity
            self.pending_slot_updates.append(('buy', time_step, dismiss_time, data_center, total_slots_needed))
            # Update server
            server_copy = copy.deepcopy(server)
            server_copy.quantity = new_quantity
            try:
                self.S.apply_server_change(server_copy)
            except ValueError as e:
                self._print(f"Error applying server change: {e}")
                return False
            self._print(f"Increased quantity of server {server_id} to {new_quantity}")
            return True

    def move_server(self):
        if not self.S.server_map:
            self._print("No servers to move")
            return False

        server = random.choice(list(self.S.server_map.values()))
        server_id = server.server_id
        time_step = server.move_info[0].time_step
        dismiss_time = server.dismiss_time
        current_datacenter = server.move_info[-1].target_datacenter
        slots_size = server.slots_size
        quantity = server.quantity

        # Randomly select a time to move
        if time_step >= dismiss_time:
            self._print("Cannot move server that has already been dismissed")
            return False
        move_time = random.randint(time_step, dismiss_time)

        # Randomly select a new datacenter
        new_datacenter = random.choice([dc for dc in self.slot_manager.total_slots.keys() if dc != current_datacenter])

        # Check slot availability in the new datacenter from move_time to dismiss_time
        if self.slot_manager.check_availability(move_time, dismiss_time, new_datacenter, slots_size * quantity):
            # Collect slot updates
            self.pending_slot_updates.append(('buy', move_time, dismiss_time, new_datacenter, slots_size * quantity))
            self.pending_slot_updates.append(('cancel', move_time, dismiss_time, current_datacenter, slots_size * quantity))
            # Update server move_info
            server_copy = copy.deepcopy(server)
            move_info = copy.deepcopy(server.move_info)
            move_info.append(ServerMoveInfo(time_step=move_time, target_datacenter=new_datacenter))
            server_copy.move_info = move_info
            try:
                self.S.apply_server_change(server_copy)
            except ValueError as e:
                self._print(f"Error applying server change: {e}")
                return False
            self._print(f"Moved server {server_id} from {current_datacenter} to {new_datacenter} at time {move_time}")
            return True
        else:
            self._print("Not enough slots in new datacenter to move server")
            return False

    def acceptance_probability(self, old_score, new_score):
        if new_score > old_score:
            return 1.0
        else:
            return np.exp((new_score - old_score) / self.current_temp)

    def run(self):
        current_score = self.evaluate()
        for i in range(self.max_iter):
            self._print(f"Iteration {i}, Temperature {self.current_temp:.2f}")
            success = self.generate_neighbor()
            if success:
                new_score = self.evaluate()
                accept_prob = self.acceptance_probability(current_score, new_score)
                if accept_prob >= 1.0 or random.random() < accept_prob:
                    # Accept the new solution
                    self.S.commit_server_changes()
                    for update in self.pending_slot_updates:
                        operation, start_time, end_time, data_center, slots_needed = update
                        self.slot_manager.update_slots(start_time, end_time, data_center, slots_needed, operation=operation)
                    current_score = new_score
                    self._print(f"Accepted new solution with score {current_score:.5e}")
                    if new_score > self.best_score:
                        self.best_solution = copy.deepcopy(self.S)
                        self.best_score = new_score
                        self._print(f"New best solution with score {self.best_score:.5e}")
                else:
                    # Revert changes
                    self.S.discard_server_changes()
                    self._print(f"Rejected new solution with score {new_score:.5e}")
            else:
                self._print("No valid neighbor found")
            self.current_temp *= self.alpha
            if self.current_temp < self.min_temp:
                break
        return self.best_solution, self.best_score

    def evaluate(self):
        return self.S.diff_evaluation()

def get_my_solution(seed, verbose=False):
    servers = pd.read_csv('./data/servers.csv')
    datacenters = pd.read_csv('./data/datacenters.csv')
    slot_manager = SlotAvailabilityManager(datacenters, verbose=verbose)
    sa = SimulatedAnnealing(slot_manager=slot_manager, seed=seed, initial_temp=100000, min_temp=100, alpha=0.999, max_iter=2500, verbose=verbose)
    best_solution, best_score = sa.run()
    # Since we are not using DataFrames anymore, we'll skip the solution export part as per your instruction
    print(f'Final best score: {best_score:.5e}')
    # Optionally, you can save or process the best_solution.server_map
    return

if __name__ == '__main__':
    start = time.time()
    seed = 3329
    get_my_solution(seed, verbose=True)
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")
