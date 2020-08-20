import astarlib
import numpy as np

map = np.load("map3.npy")
area = astarlib.aStar(map)
start = (77, 43)
goal = (77, 50)
came_from, cost = area.find_path(
    start, goal, heuristic=astarlib.euclidean_distance)
path = area.reconstruct_path(came_from, start, goal)

print(path)
print(cost)
