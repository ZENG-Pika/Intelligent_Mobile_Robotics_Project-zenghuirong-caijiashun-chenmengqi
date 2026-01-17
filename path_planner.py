"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np  # 导入 NumPy 库，主要用于处理数组格式的路径输出
import math  # 导入 math 库，用于进行平方根等数学运算
import heapq  # 导入 heapq 库，用于实现优先队列（Priority Queue），是 A* 算法的核心数据结构


class PathPlanner:
    def __init__(self, env, resolution=0.5):
        # 初始化路径规划器类
        self.env = env  # 保存传入的环境对象，用于后续调用碰撞检测函数
        self.resolution = resolution  # 设置栅格化的分辨率，决定了离散化地图的精细程度（值越小越精确，但计算量越大）

    def plan(self, start, goal):
        """
        A* Algorithm implementation.
        A* 算法的具体实现函数。

        参数:
        start: (x, y, z) tuple - 起点的实际物理坐标
        goal: (x, y, z) tuple - 终点的实际物理坐标

        返回:
        numpy array of shape (N, 3) - 包含路径点坐标的 N行3列 数组
        """
        # 将起点实际坐标转换为栅格索引坐标（离散化）
        start_node = self._to_grid(start)
        # 将终点实际坐标转换为栅格索引坐标
        goal_node = self._to_grid(goal)

        # 初始化开放列表 (OpenSet)，使用优先队列存储
        # 队列元素格式: (f_score, grid_index, current_node)
        # f_score 用于排序，越小越优先弹出
        open_set = []
        # 将起点压入优先队列，F值为0
        heapq.heappush(open_set, (0, start_node, start_node))

        # CameFrom 字典: 用于记录路径回溯
        # key: 当前节点, value: 父节点（即你是从哪个节点走到当前节点的）
        came_from = {}

        # G_score 字典: 记录从起点到当前节点的实际移动代价
        # 初始化起点的 G 值为 0
        g_score = {start_node: 0}

        # F_score 字典: 记录节点的综合评分 F = G + H
        # 初始化起点的 F 值为其启发式估算代价（H值）
        f_score = {start_node: self._heuristic(start_node, goal_node)}

        # 当开放列表不为空时，循环执行搜索
        while open_set:
            # 弹出 F 值最小的节点作为当前处理节点 (current)
            # heapq.heappop 会自动弹出元组中第一个元素（F值）最小的项
            _, _, current = heapq.heappop(open_set)

            # 判断是否到达终点
            # 如果当前节点与目标节点的距离小于分辨率，则认为已到达目标附近
            if self._dist(current, goal_node) < self.resolution:
                # 调用回溯函数，重建从起点到当前点的路径
                path = self._reconstruct_path(came_from, current, start_node)
                # 将用户指定的精确终点坐标加入路径末尾，确保终点准确无误
                path.append(goal)
                # 将列表转换为 Numpy 数组并返回，结束搜索
                return np.array(path)

            # 生成当前节点的所有邻居节点（3D空间中的26连通域）
            for neighbor in self._get_neighbors(current):
                # 碰撞检测前，先将邻居的栅格索引转换回实际物理坐标
                real_pos = self._to_real(neighbor)

                # 调用环境对象的接口进行检查
                # 如果该位置在地图边界外 (is_outside) 或与障碍物碰撞 (is_collide)，则跳过该邻居
                if self.env.is_outside(real_pos) or self.env.is_collide(real_pos):
                    continue

                # 计算经过当前节点到达邻居节点的临时 G 值
                # 新 G 值 = 当前节点 G 值 + 当前节点到邻居的距离 * 分辨率（将单位统一为物理距离）
                tentative_g_score = g_score[current] + self._dist(current, neighbor) * self.resolution

                # 如果邻居节点不在 G_score 记录中（第一次访问），或者找到了更短的路径到达该邻居
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 更新父节点记录，表示到达 neighbor 的最佳路径是经过 current
                    came_from[neighbor] = current
                    # 更新该邻居的 G 值
                    g_score[neighbor] = tentative_g_score
                    # 计算该邻居的 F 值 = 新 G 值 + 启发式 H 值
                    f = tentative_g_score + self._heuristic(neighbor, goal_node)
                    # 更新 F_score 记录
                    f_score[neighbor] = f
                    # 将该邻居加入优先队列，等待后续处理
                    heapq.heappush(open_set, (f, neighbor, neighbor))

        # 如果循环结束仍未找到路径（OpenSet 为空），打印错误信息
        print("Failed to find a path!")
        # 返回仅包含起点的数组，避免程序崩溃
        return np.array([start])

    def _heuristic(self, a, b):
        # 启发式函数 (Heuristic Function)
        # 这里使用欧几里得距离 (Euclidean Distance) 计算当前点 a 到目标点 b 的直线距离
        # 乘以 self.resolution 是为了将栅格距离转换为实际物理距离，保持单位一致
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) * self.resolution

    def _dist(self, a, b):
        # 辅助函数：计算两个栅格点之间的欧几里得距离（不带物理单位，仅数值）
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _to_grid(self, pos):
        # 坐标变换函数：将连续的物理坐标 (浮点数) 转换为离散的栅格坐标 (整数)
        # 方法是除以分辨率后四舍五入取整
        return (int(round(pos[0] / self.resolution)),
                int(round(pos[1] / self.resolution)),
                int(round(pos[2] / self.resolution)))

    def _to_real(self, grid):
        # 坐标变换函数：将离散的栅格坐标 (整数) 还原为连续的物理坐标 (浮点数)
        # 方法是乘以分辨率
        return (grid[0] * self.resolution, grid[1] * self.resolution, grid[2] * self.resolution)

    def _get_neighbors(self, node):
        # 获取邻居节点的函数
        neighbors = []
        # 遍历 x 方向的偏移 -1, 0, 1
        for dx in [-1, 0, 1]:
            # 遍历 y 方向的偏移 -1, 0, 1
            for dy in [-1, 0, 1]:
                # 遍历 z 方向的偏移 -1, 0, 1
                for dz in [-1, 0, 1]:
                    # 如果偏移量全为 0，说明是当前节点本身，跳过
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    # 将偏移量加到当前坐标上，生成邻居坐标并加入列表
                    neighbors.append((node[0] + dx, node[1] + dy, node[2] + dz))
        # 返回所有生成的邻居节点列表
        return neighbors

    def _reconstruct_path(self, came_from, current, start):
        # 路径重建函数：从终点倒推回起点
        path = []
        # 只要当前节点还在 came_from 字典中（说明还有父节点），就继续回溯
        while current in came_from:
            # 将当前栅格坐标转换为物理坐标并加入路径列表
            path.append(self._to_real(current))
            # 将当前节点更新为其父节点
            current = came_from[current]
        # 循环结束后，加入起点坐标
        path.append(self._to_real(start))
        # 因为是从终点往回推的，所以最后需要将列表反转 (`[::-1]`)
        return path[::-1]












