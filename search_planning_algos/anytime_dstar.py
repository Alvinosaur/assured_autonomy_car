import heapq
import visualizer  as viz
import numpy as np
import sys
import copy

from anytime_astar_reuse import ARA


"""
Differences from ARA:
- run backwards search since goal doesn't change, but robot position changes and we want to reuse previous path, which shouldn't change much
- account for changes to transition costs by updating the successor states of these changed paths
- specifically, addressing underconsistency by adding them back to inconsistent 
- in test, path costs are no longer just euclidean distance, but now include terrain factors, and we as robot progresses along path, certain path costs will change

"""

import heapq
import visualizer  as viz
import numpy as np
import sys
import copy


class Dstar(object):
    def __init__(self, graph):
        self.graph = graph
        # tracks expected cost to move from start to a state
        self.G = dict()  # or can be 2D array with all values init to infinity, change check of whether a state's current g-val < new g-val
        self.open_set = []
        self.going_to = {}  # backwards search, starting from goal

    def compute_path_with_reuse(self, start, goal, eps):
        closed = set()
        incons = []  # states with inconsistent values, but already expanded
        # fmin = self.open_set[0][0]  # min of minHeap is always first element
        
        # visualize order in which states are expanded
        path = {}

        i = 0
        # while len(self.open_set) > 0 and self.fgoal > fmin:
        while len(self.open_set) > 0:
            (_, current) = heapq.heappop(self.open_set)
            closed.add(current)
            path[current] = i  
            i += 1
            if current == start: break

            for next in self.graph.neighbors(current):
                # trans_cost = self.graph.cost(current, next)
                trans_cost = self.get_trans_cost(current, next)
                new_cost = self.G[current] + trans_cost
                if next not in self.G or new_cost < self.G[next]:
                    self.going_to[next] = current
                    self.G[next] = new_cost
                    h = ARA.heuristic(next, start)
                    f = ARA.compute_f(g=new_cost, h=h, eps=eps)
                    # if current == goal: self.fgoal = f
                    if next not in closed: 
                        heapq.heappush(self.open_set, (f, next))
                    else:
                        incons.append((f, next))

        # add the leftover overconsistent states from incons
        for v in incons: heapq.heappush(self.open_set, v)
        return path


    def search(self, start, goal):
        self.G = dict()
        self.G[goal] = 0  # g(goal) = 0  for backwards search
        self.open_set = [(0, goal)]  # (cost, node), PQ uses first element for priority
        self.going_to = {}
        self.going_to[goal] = None

        d_eps = 0.5
        epsilon = 2.5
        print("NOTE: Graph is literally printed, so even though G-values are\
            shown as integers, they really are floats")
        while epsilon >= 1:
            print("EPSILON: %.1f" % epsilon)
            expansions = self.compute_path_with_reuse(start, goal, epsilon)
            epsilon -= d_eps
            print("Order of expansions:")
            viz.draw_grid(self.graph, width=3, number=expansions, start=start, 
                goal=goal)
            print()
            print("G Values:")
            viz.draw_grid(self.graph, width=3, number=self.G, start=start, 
                goal=goal)
            # viz.draw_grid(viz.diagram4, width, number=cost_with_priority, start=start, goal=goal)
            print()
            print("Path:")
            viz.draw_grid(self.graph, width=3, 
                path=self.reconstruct_path(start, goal))
            
        return 

    def reconstruct_path(self, start, goal):
        current = start
        path = []
        while current != goal:
            path.append(current)
            current = self.going_to[current]
        path.append(goal)
        return path

    @staticmethod
    def heuristic(a, b):
        (x1, y1), (x2, y2) = a, b
        return abs(x2 - x1) + abs(y2 - y1)

    @staticmethod
    def compute_f(g, h, eps):
        # just to be explicit
        return g + eps*h

    @staticmethod
    def get_trans_cost(cur, next):
        (x1, y1) = cur
        (x2, y2) = next
        if int(abs(x1-x2) + abs(y1-y2)) == 1: return 1
        else: return 1.414  # sqrt(2) euclidean distance


def test():
    start = (0, 0)
    goal = (5, 6)  # (x, y)
    width = 3
    dstar = Dstar(viz.diagram3)
    dstar.search(start, goal)
    # viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
    # print()
    # viz.draw_grid(viz.diagram4, width, 
    #     path=reconstruct_path(came_from, start=start, goal=goal))


test()