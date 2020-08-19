import numpy as np
import sys
# Sample code from https://www.redblobgames.com/pathfinding/a-star/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>

# utility functions for dealing with square grids


def from_id_width(id, width):
    return (id % width, id // width)


def draw_tile(graph, id, style, width, use_np_arr):
    r = "."
    if 'number' in style:
        if use_np_arr:
            x, y = id
            num = style['number'][y, x]
            if np.isclose(num, sys.float_info.max, atol=1):
                r = "Inf"
            else:
                r = "%d" % num

        elif id in style['number']:
            r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1:
            r = "<"
        if x2 == x1 - 1:
            r = ">"
        if y2 == y1 + 1:
            r = "^"
        if y2 == y1 - 1:
            r = "V"
    if 'start' in style and id == style['start']:
        r = "S"
    if 'goal' in style and id == style['goal']:
        r = "G"
    if 'path' in style and id in style['path']:
        r = "@"
    if id in graph.walls:
        r = "#" * width
    return r


def draw_grid(graph, width=2, use_np_arr=False, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width,
                                               use_np_arr), end="")
        print()


# data from main article
DIAGRAM1_WALLS = [from_id_width(id, width=30) for id in [21, 22, 51, 52, 81, 82, 93, 94, 111, 112, 123, 124, 133, 134, 141, 142, 153, 154, 163, 164, 171, 172, 173, 174, 175,
                                                         183, 184, 193, 194, 201, 202, 203, 204, 205, 213, 214, 223, 224, 243, 244, 253, 254, 273, 274, 283, 284, 303, 304, 313, 314, 333, 334, 343, 344, 373, 374, 403, 404, 433, 434]]


class SquareGrid:
    def __init__(self, width, height, four_connected=False):
        self.width = width
        self.height = height
        self.walls = []
        self.four_connected = four_connected

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        # 8-connected grid
        (x, y) = id
        if self.four_connected:
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        else:
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1),
                       (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        if (x + y) % 2 == 0:
            results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results


class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        if (to_node == (6, 8)):
            print("SHIT")
        return self.weights.get(to_node, 0)


diagram2 = GridWithWeights(7, 7)
diagram2.walls = [(x, 4) for x in range(1, 4 + 1)]
no_go_region1 = np.zeros((7, 7))
no_go_region2 = np.zeros((7, 7))
no_go_region1[3:6, 2:4] = 5
no_go_region2[5:7, 5] = 5
total_region_costs = no_go_region1 + no_go_region2
diagram2.weights = {
    (y, x): total_region_costs[y, x] for y in range(7) for x in range(7)
}

width = 18
height = 15
diagram1 = GridWithWeights(width, height)
diagram1.walls = [(x, 13) for x in range(0, 4 + 1)]
diagram1.walls += [(x, 7) for x in range(0, 4 + 1)]
diagram1.walls += [(6, y) for y in range(14, 9 - 1, -1)]
diagram1.walls += [(7, 9), (8, 9)]
diagram1.walls += [(6, 7), (7, 7), (8, 7)]
diagram1.walls += [(6, 6)]

#  arranged y,x not x,y
total_region_costs = np.zeros(shape=(height, width))
total_region_costs[8, 6] = 10000
diagram1.weights = {
    (x, y): total_region_costs[y, x] for y in range(height) for x in range(width)
}
