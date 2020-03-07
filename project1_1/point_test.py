import numpy as np

points = np.array([
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 2.0, 0.0],
    [2.0, 2.0, 2.0],
    [0.0, 2.0, 2.0],
    [0.0, 0.0, 2.0]])
num_points = len(points)

distances = np.zeros((num_points - 1, 3))
for i in range(num_points - 1):
    distances[i] = points[i + 1] - points[i]

times = np.linalg.norm(distances,axis=1)
times = np.cumsum(times)
times = times.reshape(-1,1)
times = np.insert(times,0,0,axis = 0)

x = np.zeros((3,))
t = 0
F = 0
while F == 0:

    for ind,j in enumerate(times):
        if t == times[0]:
            x = points[0]
        elif t >= times[-1]:
            x = points[-1]
            F = 1
        elif j < t < times[ind+1]:
            x = points[ind+1]

    t += 1
    print(x)

# print(distances)
# print(times)
print(x)

# if t == 0:
#     x = points[0]
# elif 0 < t < j[0]:
#     x = points[1]
# elif j[0] < t < j[0] + 1:
#     x = points[ind + 1]
# else:
#     x = points[-1]


