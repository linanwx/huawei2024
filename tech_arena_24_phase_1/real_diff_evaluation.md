# DiffSolution 使用指南

本指南介绍如何使用 `DiffSolution` 类及相关函数来进行服务器管理和性能评估。

## 基本步骤

这个流程主要围绕创建服务器实例、应用变更、评估变更，最后决定提交或丢弃这些变更。以下是详细步骤：

### 1. 创建 `DiffSolution` 实例

首先，你需要创建一个 `DiffSolution` 实例。创建时需要提供一个随机种子 `seed` 来初始化需求数据，同时可以设置可选参数 `verbose` 为 `True` 以显示详细输出。

```python
seed = 42  # 任意随机种子
solution = DiffSolution(seed, verbose=True)
```

### 2. 创建服务器信息 (`ServerInfo`)

要表示一个服务器，你需要创建 `ServerInfo` 和 `ServerMoveInfo` 实例。`ServerMoveInfo` 实例表示服务器在特定时间步迁移到不同数据中心的情况。

```python
buy_and_move_info = [
    ServerMoveInfo(time_step=0, target_datacenter='DC1'),  # 购买到数据中心 DC1
    ServerMoveInfo(time_step=10, target_datacenter='DC2')  # 10 时间步后迁移到 DC2
]

server_info = ServerInfo(
    server_id="S1",  # 服务器 ID
    dismiss_time=100,  # 服务器下线时间
    buy_and_move_info=buy_and_move_info,  # 移动信息列表
    quantity=10,  # 服务器数量
    server_generation="CPU.S1"  # 服务器代次
)
```

### 3. 应用服务器变更

创建服务器信息后，你可以使用 `apply_server_change()` 方法将服务器变更应用到 `DiffSolution` 实例中。

```python
solution.apply_server_change(server_info)
```

### 4. 获取服务器副本

你可以使用 `get_server_copy()` 方法获取已存在服务器的深拷贝。这个方法根据 `server_id` 返回对应的 `ServerInfo` 副本，或者返回 `None` 如果该服务器不存在。

```python
server_copy = solution.get_server_copy("S1")
```

### 5. 评估变更

应用服务器变更后，你可以执行差分评估，查看变更对系统的影响。使用 `diff_evaluation()` 方法可以返回当前变更的评估结果。

```python
evaluation_result = solution.diff_evaluation()
print(f"评估结果: {evaluation_result}")
```

### 6. 提交或丢弃变更

如果对变更结果满意，你可以调用 `commit_server_changes()` 将其永久提交。如果不满意，可以通过 `discard_server_changes()` 方法丢弃变更。

- **提交变更：**

```python
solution.commit_server_changes()
```

- **丢弃变更：**

```python
solution.discard_server_changes()
```

## 示例使用

以下是一个完整的示例，展示如何创建 `DiffSolution` 实例、应用服务器变更、获取服务器副本、评估变更并提交。

```python
# 第一步：创建 DiffSolution 实例
seed = 42
solution = DiffSolution(seed, verbose=True)

# 第二步：创建服务器移动信息
buy_and_move_info = [
    ServerMoveInfo(time_step=0, target_datacenter='DC1'),
    ServerMoveInfo(time_step=10, target_datacenter='DC2')
]

# 第三步：创建服务器信息
server_info = ServerInfo(
    server_id="S1",
    dismiss_time=100,
    buy_and_move_info=buy_and_move_info,
    quantity=10,
    server_generation="CPU.S1"
)

# 第四步：应用服务器变更
solution.apply_server_change(server_info)

# 第五步：获取服务器副本
server_copy = solution.get_server_copy("S1")
print(f"服务器副本: {server_copy}")

# 第六步：评估变更
evaluation_result = solution.diff_evaluation()
print(f"评估结果: {evaluation_result}")

# 第七步：提交变更
solution.commit_server_changes()

# 如果对结果不满意，也可以选择丢弃变更
# solution.discard_server_changes()
```

## 关键方法

- `apply_server_change(diff_info)`: 应用服务器变更。
- `get_server_copy(server_id)`: 获取指定 `server_id` 的服务器副本。
- `diff_evaluation()`: 评估当前状态下的解。
- `commit_server_changes()`: 提交所应用的变更。
- `discard_server_changes()`: 丢弃当前变更，恢复之前状态。

## 结论

本指南介绍了如何使用 `DiffSolution` 类来管理服务器、应用变更、获取服务器副本和评估系统性能。通过这些方法，你可以灵活地测试不同场景并决定最终是否提交变更。