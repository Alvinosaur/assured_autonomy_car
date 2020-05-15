import heapq
import visualizer  as viz
import numpy as np
import copy

from anytime_astar_reuse import ARA_v2


"""
Differences from ARA:
- run backwards search since goal doesn't change, but robot position changes and we want to reuse previous path, which shouldn't change much
- account for changes to transition costs by updating the successor states of these changed paths
- specifically, addressing underconsistency by adding them back to inconsistent 
- in test, path costs are no longer just euclidean distance, but now include terrain factors, and we as robot progresses along path, certain path costs will change

Questions and Answers:
- 

Implementation Notes:
1. Note that all states(x,y) are Integers, which is why using them as dictionary keys works. For real-valued-states where decimal precision comes into play, need a better way to store transition costs. Maybe just a look-up table with some discretization and round to nearest state-pair.


"""


class Dstar(ARA_v2):
    def __init__(self, graph):
        super().__init__(graph)
        self.goal = None
    

    def compute_path_with_reuse(self, start, goal, eps):
        closed = set()
        incons = []  # states with inconsistent values, but already expanded
        path = {}  # visualize order in which states are expanded

        try:
            fmin = self.open_set[0][0]  # min of minHeap is always first element
        except IndexError:
            print("Nothing in OPENSET, using previous path...")
            return {}
        
        i = 0
        while len(self.open_set) > 0 and self.fgoal > fmin:
            # expand next node w/ lowest f-cost, add to closed
            (_, current) = heapq.heappop(self.open_set)
            closed.add(current)

            # visualize expansion process
            path[current] = i
            i += 1

            # V(s) = G(s), make consistent once expand
            g_cur = Dstar.get_value(self.G, current)
            Dstar.set_value(self.V, current, g_cur)
            
            # for all neighbors, check if inconsistent, if so update
            for next in self.graph.neighbors(current):
                # g(s) + c(s,s')
                trans_cost = self.get_trans_cost(current, next)
                new_cost = g_cur + trans_cost

                # if v(s) + c(s,s') < g(s'), then found better path
                if new_cost < Dstar.get_value(self.G, next):
                    # g(s') = v(s) + c(s,s')
                    Dstar.set_value(self.G, next, new_cost)

                    # store this better path to next
                    self.came_from[next] = current

                    # need to update next's neighbors, so insert updated next 
                    # into openset or incons
                    h = Dstar.heuristic(next, goal)
                    f = Dstar.compute_f(g=new_cost, h=h, eps=eps)
                    # only insert into openset if not expanded to preserve 
                    # suboptimality bound
                    if next not in closed: 
                        heapq.heappush(self.open_set, (f, next))
                    else:
                        incons.append((f, next))

                    # update fgoal if next is goal
                    if Dstar.node_equal(next, goal): 
                        self.fgoal = f
            
            # update fmin value
            if len(self.open_set) > 0:
                fmin = self.open_set[0][0]

        # add the leftover overconsistent states from incons
        for v in incons: heapq.heappush(self.open_set, v)
        return path


    def remove_from_open(self, target):
        for i in range(len(self.open_set)):
            f, node = self.open_set[i]
            if Dstar.node_equal(node, target):
                # set node to remove as last element, remove duplicate, reorder
                self.open_set[i] = self.open_set[-1]
                self.open_set.pop()  
                heapq.heapify(self.open_set)
                return

    def search(self, start, goal):
        epsilon = 1.0

        # replan from scratch
        if self.goal != goal:
            self.goal = goal
            self.G = np.ones(
                (self.graph.height, self.graph.width)) * Dstar.INF
            Dstar.set_value(self.G, start, 0)
            # need this to let computePath know when a state has an old G-value 
            # since this would otherwise not be updated
            # V-values represent idea
            # that previous G-values may need updating if edge-costs and map 
            # changes
            self.V = np.ones(
                (self.graph.height, self.graph.width)) * Dstar.INF
            
            self.open_set = [(0, start)]  # (cost, node), PQ uses first element for priority
            self.came_from = {}
            self.came_from[start] = None
            self.fgoal = Dstar.INF

            self.updated_trans_costs = {}

        # only needed for anytime D* where we iteratively decrease epsilon
        # MUST DO BEFORE THE NEXT STEP SINCE DOES NOT USE UNOBSERVABLE COSTS
        self.update_open_set(eps=epsilon, goal=goal)

        # update any edge costs as robot encounters them
        # for this simulation, only update edge costs directly reachable to robot as robot can only estimate these when close enough
        for next in self.graph.neighbors(start):
            # get transition cost, which may or may not change depending on env
            trans_cost, need_update = self.get_trans_cost(start, next, 
                observe_new_cost=True)

            # NOTE: (next, start) maps to same value, handled in get_trans_cost
            key = (start, next)  # (current, next)

            if need_update:
                # keep track of all updated transition costs
                self.updated_trans_costs[key] = trans_cost

                # if already in openset, need to remove since has outdated f-val
                self.remove_from_open(next)

                # reinsert into openset with updated f-val
                cur_g = Dstar.get_value(self.G, start)  # should be 0
                new_cost = cur_g + trans_cost  # same as trans_cost
                Dstar.set_value(self.G, next, new_cost)
                Dstar.set_value(self.V, next, Dstar.INF)
                h = Dstar.heuristic(next, goal)
                f = Dstar.compute_f(g=new_cost, h=h, eps=epsilon)
                heapq.heappush(self.open_set, (f, next))
        
        # reverse start and goal so search backwards
        expansions = self.compute_path_with_reuse(
                start=start, goal=goal, eps=epsilon)
        path = self.reconstruct_path(start=start, goal=goal)
        return path[1]  # [start, next, etc...] return next state to move to
        # return self.came_from[start]  # gives optimal successor of start node

    def get_trans_cost(self, cur, next, observe_new_cost=False):
        """Returns transition cost between two states and also simulates
        how robot can only observe true trans costs when robot is near these 
        transitions. Returns need_update=True only if we've never seen this transition(to avoid re-expanding consistent state) and robot is near enough(observe_new_cost=True) and the updated trans cost is not just the default euclidean distance.

        Arguments:
            cur {[type]} -- [description]
            next {function} -- [description]

        Keyword Arguments:
            observe_new_cost {bool} -- [description] (default: {False})

        Returns:
            [type] -- [description]
        """
        (x1, y1) = cur
        (x2, y2) = next

        # if already found true cost, simply return it, need_update=False
        key = (cur, next)
        key2 = (next, cur)  # path cost is symmetric between two states
        if key in self.updated_trans_costs:
            return self.updated_trans_costs[key], False
        if key2 in self.updated_trans_costs:
            return self.updated_trans_costs[key2], False

        # normal cost simply euclidean distance
        if int(abs(x1-x2) + abs(y1-y2)) == 1: cost = 1
        else: cost = 1.414  # sqrt(2) euclidean distance

        # If next state is actual robot position, robot can observe true path 
        # cost, so show additional, unknown cost
        need_update = False
        if observe_new_cost:
            cost += self.graph.cost(cur, next)
            # in case we've already received this updated transition cost, then 
            # don't need to re-expand since it already is consistent
            need_update = not np.isclose(
                self.graph.cost(cur, next), 0.0, atol=1e-5)

        return cost, need_update


def test():
    start = (0, 0)
    goal = (5, 6)  # (x, y)
    width = 3
    dstar = Dstar(viz.diagram2)
    while start != goal:
        next = dstar.search(start, goal)
        # make move
        start = next
        
    # viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
    # print()
    # viz.draw_grid(viz.diagram4, width, 
    #     path=reconstruct_path(came_from, start=start, goal=goal))


test()











# class Dstar(object):
#     def __init__(self, graph, viz_width=3):
#         self.graph = graph
#         self.viz_width = viz_width
#         # tracks expected cost to move from start to a state
#         self.G = dict()  # or can be 2D array with all values init to infinity, change check of whether a state's current g-val < new g-val
#         self.open_set = []
#         self.going_to = {}  # backwards search, starting from goal
#         self.fstart = sys.float_info.max

#     def compute_path_with_reuse(self, start, goal, eps):
#         closed = set()
#         incons = []  # states with inconsistent values, but already expanded
#         self.update_open_set(eps=eps, goal=start)
#         fmin = self.open_set[0][0]  # min of minHeap is always first element
        
#         # visualize order in which states are expanded
#         path = {}

#         i = 0
#         while len(self.open_set) > 0 and self.fstart > fmin:
#             (_, current) = heapq.heappop(self.open_set)
#             closed.add(current)
#             path[current] = i  
#             i += 1

#             for next in self.graph.neighbors(current):
#                 # trans_cost = self.graph.cost(current, next)
#                 trans_cost, need_update = self.get_trans_cost(current, next, start)
#                 new_cost = self.G[current] + trans_cost
#                 # if need_update:

#                 # TODO: account for the "need_update" variable above to re-update every successor state of this next state 

#                 if next not in self.G or new_cost < self.G[next]:
#                     self.going_to[next] = current
#                     self.G[next] = new_cost
#                     h = Dstar.heuristic(next, start)
#                     f = Dstar.compute_f(g=new_cost, h=h, eps=eps)
#                     if current == start: self.fstart = f
#                     if next not in closed: 
#                         heapq.heappush(self.open_set, (f, next))
#                     else:
#                         incons.append((f, next))

#         # add the leftover overconsistent states from incons
#         for v in incons: heapq.heappush(self.open_set, v)
#         return path


#     def update_open_set(self, eps, goal):
#         # call when epsilon changes and overall new f-values need to be computed
#         new_open_set = []
#         for (_, next) in self.open_set:
#             current = self.going_to[next]
#             g = self.G[next]
#             h = ARA.heuristic(next, goal)
#             f = ARA.compute_f(g=g, h=h, eps=eps)
#             heapq.heappush(new_open_set, (f, next))
        
#         self.open_set = new_open_set


#     def search(self, start, goal):
#         self.G = dict()
#         self.G[goal] = 0  # g(goal) = 0  for backwards search
#         self.open_set = [(0, goal)]  # (cost, node), PQ uses first element for priority
#         self.going_to = {}
#         self.going_to[goal] = None

#         d_eps = 0.5
#         epsilon = 2.5
#         print("NOTE: Graph is literally printed, so even though G-values are\
#             shown as integers, they really are floats")
#         while epsilon >= 1:
#             print("EPSILON: %.1f" % epsilon)
#             expansions = self.compute_path_with_reuse(start, goal, epsilon)
#             # epsilon -= d_eps
#             # print("Order of expansions:")
#             # viz.draw_grid(self.graph, width=self.viz_width, number=expansions, start=start, 
#             #     goal=goal)
#             # print()
#             # print("G Values:")
#             # viz.draw_grid(self.graph, width=self.viz_width, number=self.G, start=start, 
#             #     goal=goal)
#             # # viz.draw_grid(viz.diagram4, width, number=cost_with_priority, start=start, goal=goal)
#             # print()
#             # print("Path:")
#             # viz.draw_grid(self.graph, width=self.viz_width, 
#             #     path=self.reconstruct_path(start, goal))
            
#         return 

#     def reconstruct_path(self, start, goal):
#         current = start
#         path = []
#         while current != goal:
#             path.append(current)
#             current = self.going_to[current]
#         path.append(goal)
#         return path

#     def get_trans_cost(self, cur, next, start):
#         (x1, y1) = cur
#         (x2, y2) = next
#         (xs, ys) = start

#         # normal cost simply euclidean distance
#         if int(abs(x1-x2) + abs(y1-y2)) == 1: cost = 1
#         else: cost = 1.414  # sqrt(2) euclidean distance

#         # If next state is reachable by robot, show additional, unknown cost
#         need_update = False
#         # if int(abs(x2-xs) + abs(y2-ys)) <= 2:
#         #     cost += self.graph.cost(cur, next)
#         #     need_update = (self.graph.cost(cur, next) > 0)

#         return cost, need_update

#     @staticmethod
#     def heuristic(a, b):
#         (x1, y1), (x2, y2) = a, b
#         return abs(x2 - x1) + abs(y2 - y1)

#     @staticmethod
#     def compute_f(g, h, eps):
#         # just to be explicit
#         return g + eps*h

        



# def test():
#     start = (3, 0)
#     goal = (6, 6)  # (x, y)
#     width = 3
#     dstar = Dstar(viz.diagram2, viz_width=width)
#     # for 
#     dstar.search(start, goal)
#     # viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
#     # print()
#     # viz.draw_grid(viz.diagram4, width, 
#     #     path=reconstruct_path(came_from, start=start, goal=goal))


# test()