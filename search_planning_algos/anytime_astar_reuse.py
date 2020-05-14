"""
Key properties of Anytime-Astar:
- can return at least some solution at any given point
- improve solution over time until interrupted or converge to optimal solution, given allotted time
General approach:
- as epsilon inc, weighted A* is faster, but more greedy(so not optimal)
- initially set epsilon high to return fast solution, and with allotted time, decrease epsilon with each iter to improve solution
- include incremental A* to reuse previous computations, which should mostly stay the same
Rules of any heuristic:
- admissible: h(s) <= c*(s,s_goal): heuristic must always under-approximate true cost to goal.. if it ever over-approximates, we might not find the best path since this best path may be deemed higher cost than it should be
- also h(goal) = 0
- consistent: h(s) <= c(s,s') + h(s'): triangle ineq, heuristic directly from state s(hypotenuse) should always be less than going through another path(legs of triangle)
"""

import heapq
import visualizer  as viz
import numpy as np
import sys
import copy


class ARA(object):
    def __init__(self, graph):
        self.graph = graph
        # tracks expected cost to move from start to a state
        self.G = dict()  # or can be 2D array with all values init to infinity, change check of whether a state's current g-val < new g-val
        self.open_set = []
        self.fgoal = sys.float_info.max
        self.came_from = {}

    def compute_path_with_reuse(self, start, goal, eps):
        closed = set()
        incons = []  # states with inconsistent values, but already expanded
        self.update_open_set(eps=eps, goal=goal)
        fmin = self.open_set[0][0]  # min of minHeap is always first element
        
        # visualize order in which states are expanded
        path = {}

        i = 0
        # while len(self.open_set) > 0 and self.fgoal > fmin:
        while len(self.open_set) > 0 and self.fgoal > fmin:
            (_, current) = heapq.heappop(self.open_set)
            closed.add(current)
            path[current] = i  
            i += 1
            if current == goal: break

            for next in self.graph.neighbors(current):
                # 1 or graph.cost(current, next)
                trans_cost = ARA.get_trans_cost(current, next)
                new_cost = self.G[current] + trans_cost
                if next not in self.G or new_cost < self.G[next]:
                    self.came_from[next] = current
                    self.G[next] = new_cost
                    h = ARA.heuristic(next, goal)
                    f = ARA.compute_f(g=new_cost, h=h, eps=eps)
                    if next == goal: 
                        self.fgoal = f
                        print("FOUND GOAL!")
                        print(self.fgoal)
                    if next not in closed: 
                        heapq.heappush(self.open_set, (f, next))
                    else:
                        incons.append((f, next))
            
            # update fmin value
            if len(self.open_set) > 0:
                fmin = self.open_set[0][0]

        # add the leftover overconsistent states from incons
        for v in incons: heapq.heappush(self.open_set, v)
        return path


    def update_open_set(self, eps, goal):
        # call when epsilon changes and overall new f-values need to be computed
        new_open_set = []
        for (_, next) in self.open_set:
            current = self.came_from[next]
            g = self.G[next]
            h = ARA.heuristic(next, goal)
            f = ARA.compute_f(g=g, h=h, eps=eps)
            heapq.heappush(new_open_set, (f, next))
        
        self.open_set = new_open_set


    def search(self, start, goal):
        self.G = dict()
        self.G[start] = 0  # g(start) = 0
        self.open_set = [(0, start)]  # (cost, node), PQ uses first element for priority
        self.came_from = {}
        self.came_from[start] = None

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
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = self.came_from[current]
        path.append(start)
        path.reverse() #  reverse (goal -> start) to (start -> goal)
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
    ara = ARA(viz.diagram3)
    ara.search(start, goal)
    # viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
    # print()
    # viz.draw_grid(viz.diagram4, width, 
    #     path=reconstruct_path(came_from, start=start, goal=goal))


test()