# ---------------  USING 1 LATE DAY FOR THIS ASSIGNMENT --------------- #

import numpy as np


class WaypointTraj(object):
    """

    """

    def __init__(self, points):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission. For a waypoint
        trajectory, the input argument is an array of 3D destination
        coordinates. You are free to choose the times of arrival and the path
        taken between the points in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Inputs:
            points, (N, 3) array of N waypoint coordinates in 3D
        """


        self.points = points

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """

        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # Get distances between points
        num_points = len(self.points)
        distances = np.zeros((num_points - 1, 3))  # matrix of distances between points
        for i in range(num_points - 1):  # calculate distances between points
            distances[i] = self.points[i + 1] - self.points[i]

        # Assign times between points based on distance
        times = np.linalg.norm(distances, axis=1)
        times = np.cumsum(times)
        # times = times.reshape(-1, 1)
        times = np.insert(times, 0, 0, axis=0)  # add t = 0 to first element of time vector
        times = times / 1.2


        # Calculate quadcopter position, vel, acc based on time
        for ind, j in enumerate(times):
            if t == times[0]:  # QC at initial position at t = 0
                x = self.points[0]
                break
            elif t >= times[-1]:  # if actual time is greater than time for final position, keep QC at final position
                x = self.points[-1]
                break
            elif ind == self.points.shape[0] - 1:
                break
            elif j < t < times[ind + 1]:
                x[0], x_dot[0], x_ddot[0] = self.posvel(ind, self.points[ind], times, t, 'x') # x outputs
                x[1], x_dot[1], x_ddot[1] = self.posvel(ind, self.points[ind], times, t, 'y') # y outputs
                x[2], x_dot[2], x_ddot[2] = self.posvel(ind, self.points[ind], times, t, 'z') # z outputs

        # x_dot = 0.5 * x_dot

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

    # Calculate state using minimum acceleration or minimum jerk trajectories
    def posvel(self, ind, waypoint, times, t, pos):
        if pos == 'x':
            a = waypoint[0]
            b = self.points[ind + 1][0]
        elif pos == 'y':
            a = waypoint[1]
            b = self.points[ind + 1][1]
        elif pos == 'z':
            a = waypoint[2]
            b = self.points[ind + 1][2]

        # Minimum acceleration
        # B = np.array([[a], [b], [0], [0]])
        # T = times[ind + 1]
        # A = np.array([[1, 0, 0, 0], [1, T, T ** 2, T ** 3], [0, 1, 0, 0], [0, 1, 2 * T, 3 * (T ** 2)]])
        # X = np.linalg.solve(A, B) # calculate coefficients
        # pos = X[0] + X[1] * t + X[2] * (t ** 2) + X[3] * (t ** 3)
        # vel = X[1] + 2 * X[2] * t + 3 * X[3] * (t ** 2)

        # Minimum Jerk
        B = np.array([[a], [b], [0], [0], [0], [0]])
        T = times[ind + 1]
        A = np.array([[0, 0, 0, 0, 0, 1],
                      [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
                      [0, 0, 0, 0, 1, 0],
                      [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
                      [0, 0, 0, 2, 0, 0],
                      [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]])
        X = np.linalg.solve(A, B) # calculate coefficients
        pos = t ** 5 * X[0] + t ** 4 * X[1] + t ** 3 * X[2] + t ** 2 * X[3] + t * X[4] + X[5]
        vel = 5 * t ** 4 * X[0] + 4 * t ** 3 * X[1] + 3 * t ** 2 * X[2] + 2 * t * X[3] + X[4]
        acc = 20 * t ** 3 * X[0] + 12 * t ** 2 * X[1] + 6 * t * X[2] + 2 * X[3]

        return pos, vel, acc
