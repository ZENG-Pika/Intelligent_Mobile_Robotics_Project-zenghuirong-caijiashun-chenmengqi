"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import numpy as np  # 导入 NumPy 库，用于高效的数组操作和数学计算
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块，用于绘制结果图表
from scipy.interpolate import CubicSpline  # 从 SciPy 库导入三次样条插值类，这是轨迹生成的数学核心


class TrajectoryGenerator:
    def __init__(self, avg_speed=1.0):
        # 初始化轨迹生成器
        # avg_speed: 假设机器人的平均移动速度 (m/s)，用于将路径长度映射到时间轴
        self.avg_speed = avg_speed

    def generate_trajectory(self, path):
        """
        Generate smooth trajectory from discrete path points using Cubic Spline.
        使用三次样条插值将离散的路径点生成平滑的时间轨迹。

        参数:
        path: N x 3 numpy array - 路径规划器输出的离散路径点列表
        """
        # 确保输入路径是 NumPy 数组格式，方便切片和计算
        path = np.array(path)

        # --- 【修复部分开始】 ---
        # 预处理：删除连续的重复点或距离过近的点
        # 原因：三次样条插值要求自变量（时间 t）必须是严格单调递增的。
        # 如果有两个点坐标完全相同，它们之间的距离为0，导致对应的时间差也为0，这会引发数值计算错误。
        if len(path) > 1:
            # 初始化有效点的索引列表，默认保留第0个点（起点）
            valid_indices = [0]
            # 从第1个点开始遍历整个路径
            for i in range(1, len(path)):
                # 计算当前点 path[i] 与上一个有效保留点 path[valid_indices[-1]] 之间的欧几里得距离 (L2 Norm)
                dist = np.linalg.norm(path[i] - path[valid_indices[-1]])
                # 设置一个微小的阈值 (1mm)，只有距离大于该值的点才被认为是有效的新点
                if dist > 1e-3:
                    valid_indices.append(i)
            # 根据筛选出的索引重构路径数组
            path = path[valid_indices]
        # --- 【修复部分结束】 ---

        # 提取路径点的 x, y, z 分量
        x = path[:, 0]
        y = path[:, 1]
        z = path[:, 2]

        # 1. 计算每个路径点的时间戳 (Time Allocation)
        # 这里采用“弦长参数化”方法，即假设机器人以恒定速度沿直线运动

        # np.diff 计算相邻两个点在各轴上的差值 (dx, dy, dz)
        # 随后计算相邻点之间的欧几里得距离
        dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)

        # 防止计算出现全0导致报错的额外保护（虽然上面的预处理已经过滤了这种情况，这里作为双重保险）
        if np.any(dists == 0):
            raise ValueError("Path contains duplicate points with zero distance, which breaks spline interpolation.")

        # np.cumsum 计算累积距离，得到从起点到每个点的总路径长度
        # np.insert 在开头插入 0，代表起点的距离为 0
        cum_dist = np.insert(np.cumsum(dists), 0, 0)

        # 将距离转换为时间：时间 = 距离 / 平均速度
        # 这样我们就得到了每个路径点对应的到达时间 t_points
        t_points = cum_dist / self.avg_speed

        # 2. 使用三次样条插值 (Cubic Spline Interpolation)
        # 分别对 x(t), y(t), z(t) 进行插值，使空间曲线分解为三个关于时间的平滑函数
        # bc_type='natural' 表示使用“自然边界条件”，即轨迹起点和终点的二阶导数（加速度）为0
        self.cs_x = CubicSpline(t_points, x, bc_type='natural')
        self.cs_y = CubicSpline(t_points, y, bc_type='natural')
        self.cs_z = CubicSpline(t_points, z, bc_type='natural')

        # 保存总时长，用于后续绘图设定范围
        self.total_time = t_points[-1]
        # 保存原始路径点和对应的时间点，用于在图中绘制“叉号”标记
        self.path_points = path
        self.t_points = t_points

        # 返回轨迹的总时长
        return self.total_time

    def plot_trajectory(self):
        """
        Visualize the trajectory components x(t), y(t), z(t).
        可视化生成的轨迹，绘制 x, y, z 随时间变化的曲线。
        """
        # 生成用于绘图的密集时间点序列
        # linspace 在 0 到 总时长 之间生成 500 个均匀分布的点
        # 点越多，画出来的曲线越平滑
        t_dense = np.linspace(0, self.total_time, 500)

        # 利用之前生成的样条函数，计算这些密集时间点对应的 x, y, z 坐标
        x_smooth = self.cs_x(t_dense)
        y_smooth = self.cs_y(t_dense)
        z_smooth = self.cs_z(t_dense)

        # 创建图形窗口，包含 3 行 1 列的子图，并共享 x 轴（时间轴）
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # --- 绘制 X 轴轨迹 ---
        # 绘制平滑插值曲线 (蓝色实线)
        axs[0].plot(t_dense, x_smooth, label='Trajectory x(t)', color='blue')
        # 绘制原始路径离散点 (红色叉号)，用于对比验证插值是否经过了原点
        axs[0].plot(self.t_points, self.path_points[:, 0], 'rx', label='Path Points')
        axs[0].set_ylabel('X [m]')  # 设置 Y 轴标签
        axs[0].set_title('Trajectory Planning Results')  # 设置总标题
        axs[0].legend()  # 显示图例
        axs[0].grid(True)  # 显示网格

        # --- 绘制 Y 轴轨迹 ---
        # 绘制平滑插值曲线 (绿色实线)
        axs[1].plot(t_dense, y_smooth, label='Trajectory y(t)', color='green')
        # 绘制原始路径离散点
        axs[1].plot(self.t_points, self.path_points[:, 1], 'rx', label='Path Points')
        axs[1].set_ylabel('Y [m]')
        axs[1].legend()
        axs[1].grid(True)

        # --- 绘制 Z 轴轨迹 ---
        # 绘制平滑插值曲线 (橙色实线)
        axs[2].plot(t_dense, z_smooth, label='Trajectory z(t)', color='orange')
        # 绘制原始路径离散点
        axs[2].plot(self.t_points, self.path_points[:, 2], 'rx', label='Path Points')
        axs[2].set_ylabel('Z [m]')
        axs[2].set_xlabel('Time [s]')  # 设置 X 轴标签（仅在最底部的图显示）
        axs[2].legend()
        axs[2].grid(True)

        # 自动调整子图间距，防止标签重叠
        plt.tight_layout()
        # 显示图像
        plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
#
#
# class TrajectoryGenerator:
#     def __init__(self, avg_speed=1.0):
#         self.avg_speed = avg_speed
#
#     def generate_trajectory(self, path):
#         """
#         Generate smooth trajectory from discrete path points using Cubic Spline.
#         path: N x 3 numpy array
#         """
#         path = np.array(path)
#         x = path[:, 0]
#         y = path[:, 1]
#         z = path[:, 2]
#
#         # 1. 计算每个路径点的时间戳 (Time Allocation)
#         # 计算相邻点之间的距离
#         dists = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
#         # 累积距离得到总距离
#         cum_dist = np.insert(np.cumsum(dists), 0, 0)
#         # 时间 = 距离 / 速度
#         t_points = cum_dist / self.avg_speed
#
#         # 2. 使用三次样条插值 (Cubic Spline Interpolation)
#         # bc_type='natural' 表示两端二阶导数为0 (自然样条)
#         self.cs_x = CubicSpline(t_points, x, bc_type='natural')
#         self.cs_y = CubicSpline(t_points, y, bc_type='natural')
#         self.cs_z = CubicSpline(t_points, z, bc_type='natural')
#
#         self.total_time = t_points[-1]
#         self.path_points = path
#         self.t_points = t_points
#
#         return self.total_time
#
#     def plot_trajectory(self):
#         """
#         Visualize the trajectory components x(t), y(t), z(t).
#         """
#         # 生成密集的时间点用于绘图平滑曲线
#         t_dense = np.linspace(0, self.total_time, 500)
#
#         x_smooth = self.cs_x(t_dense)
#         y_smooth = self.cs_y(t_dense)
#         z_smooth = self.cs_z(t_dense)
#
#         fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
#
#         # Plot X
#         axs[0].plot(t_dense, x_smooth, label='Trajectory x(t)', color='blue')
#         axs[0].plot(self.t_points, self.path_points[:, 0], 'rx', label='Path Points')
#         axs[0].set_ylabel('X [m]')
#         axs[0].set_title('Trajectory Planning Results')
#         axs[0].legend()
#         axs[0].grid(True)
#
#         # Plot Y
#         axs[1].plot(t_dense, y_smooth, label='Trajectory y(t)', color='green')
#         axs[1].plot(self.t_points, self.path_points[:, 1], 'rx', label='Path Points')
#         axs[1].set_ylabel('Y [m]')
#         axs[1].legend()
#         axs[1].grid(True)
#
#         # Plot Z
#         axs[2].plot(t_dense, z_smooth, label='Trajectory z(t)', color='orange')
#         axs[2].plot(self.t_points, self.path_points[:, 2], 'rx', label='Path Points')
#         axs[2].set_ylabel('Z [m]')
#         axs[2].set_xlabel('Time [s]')
#         axs[2].legend()
#         axs[2].grid(True)
#
#         plt.tight_layout()
#         plt.show()