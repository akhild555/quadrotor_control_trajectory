import numpy as np

from proj1_3.code.graph_search import graph_search


class WorldTraj(object):
    """

    """

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = self.sparsePoints(self.path)  # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

    # def sparsePoints(self, path):
    #     linPoints = []
    #     for ind, i in enumerate(path):
    #         if ind == path.shape[0] - 1:
    #             break
    #         else:
    #             diff = i - path[ind + 1]
    #             if sum(diff == 0) == 2 or sum(diff) == 1:
    #                 linPoints.append(ind)
    #     path = np.delete(path,linPoints[::2],0)
    #     return path

    # Remove points whose vectors are parallel to each other using cross product
    def sparsePoints(self, path):
        linPoints = []
        for ind, i in enumerate(path):
            if ind == path.shape[0] - 1:
                break
            else:
                if ind == 0:
                    check = np.ones(3,)
                else:
                    check = np.cross(path[ind-1]-i,path[ind+1]-i)
                if np.array_equal(check, np.zeros(3,)):
                    linPoints.append(ind)
        linPoints = linPoints[::2]
        path = np.delete(path,linPoints,0)
        return path



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

        # STUDENT CODE HERE

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}

        num_points = len(self.points)
        distances = np.zeros((num_points - 1, 3))  # matrix of distances between points
        for i in range(num_points - 1):  # calculate distances between points
            distances[i] = self.points[i + 1] - self.points[i]

        times = np.linalg.norm(distances, axis=1)  # calculate times btwn points based on distance
        times = np.cumsum(times)
        # times = times.reshape(-1, 1)
        times = np.insert(times, 0, 0, axis=0)  # add t = 0 to first element of time vector
        times = times / 3.0

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
                x[0], x_dot[0], x_ddot[0] = self.posvel(ind, self.points[ind], times, t, flat_output, 'x')
                x[1], x_dot[1], x_ddot[1] = self.posvel(ind, self.points[ind], times, t, flat_output, 'y')
                x[2], x_dot[2], x_ddot[2] = self.posvel(ind, self.points[ind], times, t, flat_output, 'z')
                # x[0], x_dot[0] = self.posvel(ind, self.points[ind], times, t, flat_output, 'x')
                # x[1], x_dot[1] = self.posvel(ind, self.points[ind], times, t, flat_output, 'y')
                # x[2], x_dot[2] = self.posvel(ind, self.points[ind], times, t, flat_output, 'z')

        x_ddot = x_ddot.clip(max=3)
        x_ddot = x_ddot.clip(min=-3)
        x_dot = x_dot.clip(max=2)
        x_dot = x_dot.clip(min=-2)


        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

    def posvel(self, ind, waypoint, times, t, flat, pos):
        if pos == 'x':
            a = waypoint[0]
            b = self.points[ind + 1][0]
            c =  flat['x_dot'][0]
            d =  flat['x_dot'][0]
            e =  flat['x_ddot'][0]
            f =  flat['x_ddot'][0]
        elif pos == 'y':
            a = waypoint[1]
            b = self.points[ind + 1][1]
            c = flat['x_dot'][1]
            d = flat['x_dot'][1]
            e = flat['x_ddot'][1]
            f = flat['x_ddot'][1]
        elif pos == 'z':
            a = waypoint[2]
            b = self.points[ind + 1][2]
            c = flat['x_dot'][2]
            d = flat['x_dot'][2]
            e = flat['x_ddot'][2]
            f = flat['x_ddot'][2]

        # Minimum acceleration
        # B = np.array([[a], [b], [0], [0]])
        # T = times[ind + 1]
        # A = np.array([[1, 0, 0, 0], [1, T, T ** 2, T ** 3], [0, 1, 0, 0], [0, 1, 2 * T, 3 * (T ** 2)]])
        # X = np.linalg.solve(A, B)
        # pos = X[0] + X[1] * t + X[2] * (t ** 2) + X[3] * (t ** 3)
        # vel = X[1] + 2 * X[2] * t + 3 * X[3] * (t ** 2)
        # # acc = 2 * X[2] + 6 * X[3] * t

        # Minimum Jerk
        B = np.array([[a], [b], [c], [d], [e], [f]])
        T = times[ind + 1]
        A = np.array([[0, 0, 0, 0, 0, 1],
                      [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
                      [0, 0, 0, 0, 1, 0],
                      [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
                      [0, 0, 0, 2, 0, 0],
                      [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0]])
        X = np.linalg.solve(A, B)
        pos = t ** 5 * X[0] + t ** 4 * X[1] + t ** 3 * X[2] + t ** 2 * X[3] + t * X[4] + X[5]
        vel = 5 * t ** 4 * X[0] + 4 * t ** 3 * X[1] + 3 * t ** 2 * X[2] + 2 * t * X[3] + X[4]
        acc = 20 * t ** 3 * X[0] + 12 * t ** 2 * X[1] + 6 * t * X[2] + 2 * X[3]
        # jerk = 60 * t ** 2 * X[0] + 24 * t * X[1] + 6 * X[2]

        return pos, vel, acc