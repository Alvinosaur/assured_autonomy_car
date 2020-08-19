# The MIT License
#
# Copyright (c) 2019 Herbert Shin  https://github.com/initbar/astarlib
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
astarlib
--------

This module implements A* pathfinding algorithm for graph and 2d search space.
"""

from collections import deque
from libc.math cimport pow
from libc.math cimport sqrt
import heapq

__all__ = (
    "NEIGHBOR_LINEAR_SQUARE",
    "NEIGHBOR_STAR_SQUARE",
    "PathNotFoundException",
    "aStar",
)


#
# Exceptions
#

class PathNotFoundException(ValueError):
    """A* path to destination not found"""
    pass


#
# Heuristics
#

# +---+---+---+
# | X | X | X |
# +---+---+---+
# | X |   | X |
# +---+---+---+
# | X | X | X |
# +---+---+---+
NEIGHBOR_STAR_SQUARE = (  # delta table for linear and diagonal neighbors
    (-1, 1),  (0, 1),  (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1),
)

# +---+---+---+
# |   | X |   |
# +---+---+---+
# | X |   | X |
# +---+---+---+
# |   | X |   |
# +---+---+---+
NEIGHBOR_LINEAR_SQUARE = (  # delta table for linear neighbors
    (0, 1), (-1, 0), (1, 0), (0, -1),
)

cpdef float euclidean_distance(int x1, int y1, int x2, int y2):
    """Euclidean distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    |   |   | E |
    +---+---/---+
    |   | / |   |
    +---/---+---+
    | S |   |   |  distance S->E = 2
    +---+---+---+
    """
    return sqrt(pow(x2 - x1, 2) + pow(y1 - y2, 2))

cpdef unsigned int chebyshev_distance(int x1, int y1, int x2, int y2):
    """Chebyshev distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    | E | E | E |
    +---\-|-/---+
    | E - S - E |
    +---/-|-\---+
    | E | E | E |  distance S->E = 1
    +---+---+---+
    """
    return max(abs(x2 - x1), abs(y2 - y1))

cpdef unsigned int manhattan_distance(int x1, int y1, int x2, int y2):
    """Manhattan distance between two points: S (x1, y1) and E (x2, y2).

    Example:
    +---+---+---+
    | E | E | E |
    +---\-|-/---+
    | E - S - E |
    +---/-|-\---+
    | E | E | E |  distance S->E = {1, 2}
    +---+---+---+
    """
    return abs(x1 - x2) + abs(y1 - y2)


#
# A*
#

cdef class pNode:

    cdef public int x, y
    cdef public float g, h, f

    # unlike conventional graphs or trees, the `parent` and `child`
    # relationship here is used in the context of a linked-list.
    cdef public object parent, child

    def __cinit__(self, pos, g=0, h=0, parent=None, child=None):
        """represents point-node (pNode).
        :type x: int or float
        :param x: value on Cartesian x-axis.
        :type y: int or float
        :param y: value on Cartesian y-ayis.
        :type parent: pNode
        :param parent: parent pNode.
        :type child: pNode
        :param child: child pNode.
        """
        self.x, self.y = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.child = child

    def __lt__(self, other):
        return self.f < other.f

    @property
    def point(self):
        return (self.x, self.y)  # (x, y)

cdef resolve_child_pnodes_to_points(pnode):
    """resolve linked-list pNodes into sequence of points in head to tail direction.
    :type pnode: pNode
    :param pnode: pNode instance.

    This function returns (a.point .. z.point):

    +----------+     +---+            +----------+
    | pNode: a | --> | b | --> .. --> | pNode: z |
    +----------+     +---+            +----------+
       (head)                            (tail)
    """
    head = pnode
    if head.child is None:
        return (head.point,)
    path = [head.point]
    while head.child is not None:
        path.append(head.child.point)
        head = head.child
    return tuple(path)

cdef resolve_parent_pnodes_to_points(pnode, reverse=False):
    """resolve linked-list pNodes into sequence of points in tail to head direction.
    :type pnode: pNode
    :param pnode: pNode instance.

    :type reverse: bool
    :param reverse: reverse the resolution order.

    This function returns (a.point .. z.point) or (z.point .. a.point)
    depending on the `reverse` state:

    +----------+     +---+            +----------+
    | pNode: a | <-- | b | <-- .. <-- | pNode: z |
    +----------+     +---+            +----------+
       (head)                            (tail)
    """
    tail = pnode
    if tail.parent is None:
        return (tail.point,)
    # utilize double-ended queue to pre-inject the resolved points in sequential
    # or reversed direction. Otherwise, we need to unnecessarily iterate through
    # the sequence twice by calling [::-1] on the result.
    path = deque([tail.point])
    while tail.parent is not None:
        if reverse is True:
            path.appendleft(tail.parent.point)  # (a.point -> z.point)
        else:
            path.append(tail.parent.point)  # (z.point -> a.point)
        tail = tail.parent
    return tuple(path)


class aStar(object):

    __slots__ = "_buffer", "_height", "_width"

    def __init__(self, array=[[]]):
        self._height = len(array)
        self._width = len(array[-1])
        self._buffer = array

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width
        
    def find_path(self, start, goal, 
                        heuristic=euclidean_distance):
        """find A* path from `start` to `end`. Obstacles denoted in binary map as 1 and free space as 0.
        :type start: tuple
        :param start: starting position in (x, y).

        :type end: tuple
        :param end: destination position in (x, y).

        :type heuristic: function
        :param heuristic: heuristic function to calculate H(n).
        """
        sx, sy = start
        gx, gy = goal
        frontier = []
        # initial h value doesn't matter
        heapq.heappush(frontier, pNode(start, h=0))
        came_from = {start: None}
        closed = set()
        cost_so_far = {start: 0}
        i = 0
        while len(frontier) > 0:
            current_node = heapq.heappop(frontier)
            current_cost = cost_so_far[current_node.point]
            closed.add(current_node.point)
            if current_node.point == goal:
                return came_from, current_node.g  # travel cost is cost-to-go
            for next in self.neighbors_at(x=current_node.x, y=current_node.y):
                nx, ny = next
                if next in closed:
                    continue
                elif self._buffer[ny][nx] is True:
                    # ignore this neighbor (an obstacle).
                    closed.add(next)
                    continue
                
                new_cost = current_cost + self.get_trans_cost(current_node.point, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    heapq.heappush(frontier, pNode(next, g=new_cost, h=heuristic(x1=nx, y1=ny, x2=gx, y2=gy)))
                    came_from[next] = current_node.point

        raise PathNotFoundException

    def neighbors_at(self, x, y, delta=NEIGHBOR_STAR_SQUARE):
        """get indexes of neighboring elements of (x, y)"""
        if not delta:
            yield None
        # traverse through immediately adjacent elements and ignore non-existent
        # neighbors. For example, no valid neighbors exist adjacent to (-2, -2),
        # but (0, -1) is adjacent to a valid neighbor (0, 0).
        cdef int nx, ny
        for dx, dy in delta:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    @staticmethod
    def get_trans_cost(s1, s2):
        (x1, y1) = s1
        (x2, y2) = s2
        if int(abs(x1 - x2) + abs(y1 - y2)) == 1:
            return 1
        else:
            return 1.4142135623730951  # sqrt(2)

    @staticmethod
    def reconstruct_path(came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()  # reverse (goal -> start) to (start -> goal)
        return path
