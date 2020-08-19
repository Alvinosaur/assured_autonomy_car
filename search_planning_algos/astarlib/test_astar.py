from astarlib import aStar, euclidean_distance
import numpy as np

map = np.load("map3.npy")
area = aStar(map)
start = (60, 40)
goal = (85, 65)
path, cost = area.find_path(start, goal, heuristic=euclidean_distance)

print(path)
print(cost)
