class SAParameterGenerator:
    def __init__(self, start_value, end_value, total_steps, rule='linear'):
        """
        初始化参数生成器
        :param start_value: 参数的起始值
        :param end_value: 参数的终止值
        :param total_steps: 模拟退火的最大步骤数
        :param rule: 参数变化的规则 ('linear', 'exponential', 'custom')
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.rule = rule
    
    def get_parameter(self, current_step):
        """
        根据当前步骤返回参数值
        :param current_step: 当前步骤
        :return: 当前参数值
        """
        if self.rule == 'linear':
            return self._linear_interpolation(current_step)
        elif self.rule == 'exponential':
            return self._exponential_interpolation(current_step)
        # 可以添加其他的规则处理逻辑
        else:
            raise ValueError(f"Unsupported rule: {self.rule}")
    
    def _linear_interpolation(self, current_step):
        """
        线性变化
        """
        return self.start_value + (self.end_value - self.start_value) * (current_step / self.total_steps)
    
    def _exponential_interpolation(self, current_step):
        """
        指数变化
        """
        return self.start_value * ((self.end_value / self.start_value) ** (current_step / self.total_steps))