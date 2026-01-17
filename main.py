from flight_environment import FlightEnvironment
from path_planner import PathPlanner         # 导入你的路径规划器
from trajectory_generator import TrajectoryGenerator # 导入你的轨迹生成器
import numpy as np

# 初始化环境，随机种子如果不设置，每次运行障碍物位置都会变
# 建议调试时可以在 FlightEnvironment 内部固定 random seed
env = FlightEnvironment(50)
start = (1,2,0)
goal = (18,18,3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# path = [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
print("Planning Path...")
planner = PathPlanner(env, resolution=0.5) # 实例化
path = planner.plan(start, goal)           # 执行规划

# --------------------------------------------------------------------------------------------------- #

# 这里的 path 已经是 numpy array 了，直接传给 env 绘图
env.plot_cylinders(path)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.
print("Generating Trajectory...")
traj_gen = TrajectoryGenerator(avg_speed=2.0) # 假设速度 2m/s
traj_gen.generate_trajectory(path)

# Plot the trajectory (x-t, y-t, z-t)
traj_gen.plot_trajectory()
print("Done! Check the figures.")


# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
