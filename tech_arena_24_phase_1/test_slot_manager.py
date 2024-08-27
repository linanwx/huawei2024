import unittest
import pandas as pd
import numpy as np

from SA import SlotAvailabilityManager

class TestSlotAvailabilityManager(unittest.TestCase):
    def setUp(self):
        # 构建 datacenters 数据
        datacenters_data = {
            'datacenter_id': ['DC1', 'DC2', 'DC3', 'DC4'],
            'cost_of_energy': [0.25, 0.35, 0.65, 0.75],
            'latency_sensitivity': ['low', 'medium', 'high', 'high'],
            'slots_capacity': [25245, 15300, 7020, 8280]
        }
        self.datacenters = pd.DataFrame(datacenters_data)

        # 初始化 SlotAvailabilityManager
        self.slot_manager = SlotAvailabilityManager(self.datacenters)

    def test_initial_slots(self):
        # 测试初始化插槽容量是否正确
        self.assertEqual(self.slot_manager.slots_table.at[0, 'DC1'], 25245)
        self.assertEqual(self.slot_manager.slots_table.at[1, 'DC2'], 15300)
        self.assertEqual(self.slot_manager.slots_table.at[2, 'DC3'], 7020)
        self.assertEqual(self.slot_manager.slots_table.at[167, 'DC4'], 8280)

    def test_check_availability(self):
        # 测试插槽可用性检查
        available = self.slot_manager.check_availability(1, 96, 'DC1', 100)
        self.assertTrue(available)

        print(self.slot_manager.slots_table)
        # 模拟插槽占用，再次检查
        self.slot_manager.update_slots(1, 96, 'DC1', 100, operation='buy')
        available = self.slot_manager.check_availability(1, 96, 'DC1', 25246)
        self.assertFalse(available)
        print(self.slot_manager.slots_table)

    def test_update_slots(self):
        # 测试插槽更新
        self.slot_manager.update_slots(1, 96, 'DC1', 100, operation='buy')
        for ts in range(1, 97):
            self.assertEqual(self.slot_manager.slots_table.at[ts-1, 'DC1'], 25245 - 100)

        # 撤销操作，再次检查
        self.slot_manager.update_slots(1, 96, 'DC1', 100, operation='cancel')
        for ts in range(1, 97):
            self.assertEqual(self.slot_manager.slots_table.at[ts-1, 'DC1'], 25245)

    def test_no_slot_leak(self):
        # 测试在整个生命周期内是否正确占用插槽
        self.slot_manager.update_slots(1, 96, 'DC1', 100, operation='buy')
        self.slot_manager.update_slots(97, 168, 'DC1', 200, operation='buy')
        for ts in range(1, 97):
            self.assertEqual(self.slot_manager.slots_table.at[ts-1, 'DC1'], 25145)
        for ts in range(97, 169):
            self.assertEqual(self.slot_manager.slots_table.at[ts-1, 'DC1'], 25045)

if __name__ == '__main__':
    unittest.main()
