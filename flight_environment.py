import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from mpl_toolkits.mplot3d import Axes3D

class FlightEnvironment:
    def __init__(self,obs_num):
        self.env_width = 20.0
        self.env_length = 20.0
        self.env_height = 5
        self.space_size = (self.env_width,self.env_length,self.env_height)
        self._obs_num = obs_num

        self.cylinders = self.generate_random_cylinders(self.space_size,self._obs_num,0.1,0.3,5,5)




    def generate_random_cylinders(self,space_size, N,
                              min_radius, max_radius,
                              min_height, max_height,
                              max_tries=100000):

        X, Y, Z = space_size
        cylinders = []
        tries = 0

        while len(cylinders) < N and tries < max_tries:
            tries += 1

            r = np.random.uniform(min_radius, max_radius)
            h = np.random.uniform(min_height, min(max_height, Z))

            x = np.random.uniform(r, X - r)
            y = np.random.uniform(r, Y - r)

            candidate = np.array([x, y, h, r])

            no_overlapping = True
            for c in cylinders:
                dx = x - c[0]
                dy = y - c[1]
                dist = np.hypot(dx, dy)
                if dist < (r + c[3]):  
                    no_overlapping = False
                    break

            if no_overlapping:
                cylinders.append(candidate)

        if len(cylinders) < N:
            raise RuntimeError("Unable to generate a sufficient number of non-overlapping cylinders with the given parameters. Please reduce N or decrease the radius range.")

        return np.vstack(cylinders)
    

    def is_outside(self,point):
        """
        Check whether a 3D point lies outside the environment boundary.

        Parameters:
            point : tuple or list (x, y, z)
                The coordinates of the point to be checked.

        Returns:
            bool
                True  -> the point is outside the environment limits  
                False -> the point is within the valid environment region
        """

        x,y,z = point
        if (0 <= x <= self.env_width and
                0 <= y <= self.env_length and
                0 <= z <= self.env_height):
            outside_env = False
        else:
            outside_env = True
        return outside_env
    


    def is_collide(self, point, epsilon=0.2):
            """
            Check whether a point in 3D space collides with a given set of cylinders (including a safety margin).

            Parameters:
                point: A numpy array or tuple of (x, y, z)
                cylinders: An N×4 numpy array, each row is [cx, cy, h, r]
                        where cx, cy are the cylinder center coordinates in XY,
                        h is the height, and r is the radius
                epsilon: Safety margin; if the point is closer than (r + epsilon),
                        it is also considered a collision

            Returns:
                True  -> Collision (or too close)
                False -> Safe
            """
            cylinders = self.cylinders
            px, py, pz = point

            for cx, cy, h, r in cylinders:
                if not (0 <= pz <= h):
                    continue 
                dist_xy = np.sqrt((px - cx)**2 + (py - cy)**2)
                if dist_xy <= (r + epsilon):
                    return True   
            
            return False
    
    def plot_cylinders(self,path = None):
        """
        cylinders: N×4 array, [cx, cy, h, r]
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cylinders = self.cylinders
        space_size = self.space_size



        Xmax, Ymax, Zmax = space_size
        for cx, cy, h, r in cylinders:
            z = np.linspace(0, h, 30)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta, z = np.meshgrid(theta, z)

            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)

            ax.plot_surface(x, y, z, color='skyblue', alpha=0.8)
            theta2 = np.linspace(0, 2*np.pi, 30)
            x_top = cx + r * np.cos(theta2)
            y_top = cy + r * np.sin(theta2)
            z_top = np.ones_like(theta2) * h
            ax.plot_trisurf(x_top, y_top, z_top, color='steelblue', alpha=0.8)

        ax.set_xlim(0, self.env_width)
        ax.set_ylim(0, self.env_length)
        ax.set_zlim(0, self.env_height)


        if path is not None:
            path = np.array(path)
            xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
            ax.plot(xs, ys, zs, linewidth=2)     
            ax.scatter(xs[0], ys[0], zs[0], s=40) 
            ax.scatter(xs[-1], ys[-1], zs[-1], s=40) 
        self.set_axes_equal(ax)
        plt.show()


    def set_axes_equal(self,ax):
        """Make axes of 3D plot have equal scale.
        Compatible with Matplotlib ≥ 1.0.0
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        max_range = max([x_range, y_range, z_range]) / 2.0

        mid_x = (x_limits[0] + x_limits[1]) * 0.5
        mid_y = (y_limits[0] + y_limits[1]) * 0.5
        mid_z = (z_limits[0] + z_limits[1]) * 0.5

        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

