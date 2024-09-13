import unittest
import numpy as np
from real_diff_evaluation import DiffSolution, ServerInfo, DiffBlackboard

class TestDiffSolution(unittest.TestCase):
    def setUp(self):
        self.diff_solution = DiffSolution()
        self.blackboard = DiffBlackboard(lifespan=np.zeros(168, dtype=int))

    def test_apply_change_basic(self):
        diff_info = ServerInfo(
            server_id="server_1",
            buy_time=10,
            dismiss_time=20,
            datacenter_id="dc_1",
            move_info=[]
        )
        self.diff_solution._apply_change(diff_info, self.blackboard)
        print(self.blackboard)
        expected_lifespan = np.zeros(168, dtype=int)
        expected_lifespan[10:20] = np.arange(1, 11)
        np.testing.assert_array_equal(self.blackboard.lifespan, expected_lifespan)

    # def test_apply_change_with_initial_lifespan(self):
    #     diff_info = ServerDiffInfo(
    #         server_id="server_2",
    #         buy_time=5,
    #         dismiss_time=15,
    #         datacenter_id="dc_2",
    #         move_info=[]
    #     )
    #     self.diff_solution.lifespanManager.update_lifespan(5, 15, initial_lifespan=2)
    #     self.diff_solution._apply_change(diff_info, self.blackboard)
    #     expected_lifespan = np.zeros(168, dtype=int)
    #     expected_lifespan[5:15] = np.arange(2, 12)
    #     np.testing.assert_array_equal(self.blackboard.lifespan, expected_lifespan)

    # def test_apply_change_with_existing_lifespan(self):
    #     self.blackboard.lifespan[0:10] = np.arange(1, 11)
    #     diff_info = ServerDiffInfo(
    #         server_id="server_3",
    #         buy_time=0,
    #         dismiss_time=10,
    #         datacenter_id="dc_3",
    #         move_info=[]
    #     )
    #     self.diff_solution._apply_change(diff_info, self.blackboard)
    #     expected_lifespan = np.arange(2, 12)
    #     np.testing.assert_array_equal(self.blackboard.lifespan, expected_lifespan)

if __name__ == '__main__':
    unittest.main()