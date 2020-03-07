import heapq
from heapq import heappush, heappop  # Recommended.

import numpy as np

from flightsim.world import World
from proj1_2.code.occupancy_map import OccupancyMap  # Recommended.


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    visited_nodes = []
    cost2come = np.inf * np.ones_like(occ_map.map)
    parent_nodes = 0 * np.empty_like(occ_map.map)
    parent_nodes = parent_nodes.tolist()

    if astar:
        pHeap = [(0, 0, start_index)]
    else:
        pHeap = [(0, start_index)]

    # List all neighbors (26-connectivity)
    neighbors = [[-1, 0, 1], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [0, 0, -1], [1, 0, -1],
                          [-1, 1, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 0], [1, 1, 0], [-1, 1, -1], [0, 1, -1], [1, 1, -1],
                          [-1, -1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 0], [1, -1, 0], [-1, -1, -1], [0, -1, -1],
                          [1, -1, -1],
                          [0, -1, 0], [0, 1, 0]]

    while len(pHeap) > 0:

        if astar:
            f, cur_c2c, cur_index = heappop(pHeap)
        else:
            cur_c2c, cur_index = heappop(pHeap)

        if goal_index in visited_nodes: # exit if goal is visited
            break
        if not occ_map.is_valid_index(cur_index): # skip node if outside map
            continue
        if cur_index in visited_nodes: # skip node if already visited
            continue
        if occ_map.is_occupied_index(cur_index): # skip node if there's an obstacle there
            continue
        for i in neighbors:
            i = np.array(i)
            neighbor_ind = cur_index + i
            if not occ_map.is_valid_index(neighbor_ind): # skip neighbor if outside map
                continue
            if occ_map.is_occupied_index(neighbor_ind): # skip neighbor if occupied by obstacle
                continue

            if astar:
                dist = np.linalg.norm(i)
                c2c = cur_c2c + dist
                h = np.linalg.norm(np.array((cur_index))-np.array((goal_index)))**1.5 # heuristic (L2 Norm)
                f = c2c + h  # f = g + h
            else:
                dist = np.linalg.norm(i)
                c2c = cur_c2c + dist

            if c2c < cost2come[(neighbor_ind)[0], (neighbor_ind)[1], (neighbor_ind)[2]]:
                cost2come[(neighbor_ind)[0], (neighbor_ind)[1], (neighbor_ind)[2]] = c2c
                if astar:
                    heappush(pHeap, [f, c2c, tuple(neighbor_ind)])
                else:
                    heappush(pHeap, [c2c, tuple(neighbor_ind)])
                parent_nodes[(neighbor_ind)[0]][(neighbor_ind)[1]][(neighbor_ind)[2]] = cur_index

        visited_nodes.append(tuple(cur_index))

    if parent_nodes[goal_index[0]][goal_index[1]][goal_index[2]] == 0:
        return None

    # build path from start to goal using parent_nodes
    backTostart = np.array([])
    path = np.array(goal)
    counter = 0
    while not np.array_equal(backTostart,np.array(start_index)):
        if counter == 0:
            backTostart = parent_nodes[goal_index[0]][goal_index[1]][goal_index[2]]
        else:
            backTostart = parent_nodes[backTostart[0]][backTostart[1]][backTostart[2]]
        path = np.vstack((path,occ_map.index_to_metric_center(backTostart)))

        counter += 1
    path = np.vstack((path, start))
    path = np.flip(path,axis=0)

    return path
