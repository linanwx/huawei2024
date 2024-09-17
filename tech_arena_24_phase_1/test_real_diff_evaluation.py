import unittest
from unittest.mock import MagicMock
from real_diff_evaluation import DiffSolution, ServerInfo, ServerMoveInfo

class TestDiffSolution(unittest.TestCase):

    def setUp(self):
        self.diff_solution = DiffSolution(123)
        self.diff_solution.server_counts = MagicMock()
        self.diff_solution._handle_server_reduction = MagicMock()
        self.diff_solution._finalize_dismiss_times = MagicMock()
        self.diff_solution._can_move_to_other_dc = MagicMock()
        self.diff_solution._find_new_dc = MagicMock()
        global TIME_STEPS, datacenter_info_dict, SERVER_GENERATION_MAP
        TIME_STEPS = 10
        datacenter_info_dict = {'dc1': {}, 'dc2': {}}
        SERVER_GENERATION_MAP = {'gen1': 0, 'gen2': 1}

    def test_construct_server_map_initial_purchase(self):
        self.diff_solution.server_counts = [
            [[5, 0], [0, 0]],  # t=0
            [[5, 0], [0, 0]],  # t=1
        ]
        server_map = self.diff_solution.construct_server_map()
        self.assertEqual(len(server_map), 1)
        self.assertEqual(server_map['S1'].quantity, 5)
        self.assertEqual(server_map['S1'].server_generation, 'gen1')

    # def test_construct_server_map_additional_purchase(self):
    #     self.diff_solution.server_counts = [
    #         [[5, 0], [0, 0]],  # t=0
    #         [[10, 0], [0, 0]],  # t=1
    #     ]
    #     server_map = self.diff_solution.construct_server_map()
    #     self.assertEqual(len(server_map), 2)
    #     self.assertEqual(server_map['S1'].quantity, 5)
    #     self.assertEqual(server_map['S2'].quantity, 5)

    # def test_construct_server_map_server_reduction(self):
    #     self.diff_solution.server_counts = [
    #         [[5, 0], [0, 0]],  # t=0
    #         [[3, 0], [0, 0]],  # t=1
    #     ]
    #     self.diff_solution.construct_server_map()
    #     self.diff_solution._handle_server_reduction.assert_called_once_with(
    #         unittest.mock.ANY, 1, 'dc1', 'gen1', 2
    #     )

    # def test_construct_server_map_finalize_dismiss_times(self):
    #     self.diff_solution.server_counts = [
    #         [[5, 0], [0, 0]],  # t=0
    #         [[5, 0], [0, 0]],  # t=1
    #     ]
    #     self.diff_solution.construct_server_map()
    #     self.diff_solution._finalize_dismiss_times.assert_called_once()

if __name__ == '__main__':
    unittest.main()