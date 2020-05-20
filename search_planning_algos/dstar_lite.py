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

DEBUG = False

class Node(object):
    def __init__(self, k1, k2, state):
        self.k1 = k1
        self.k2 = k2
        self.state = state

    def __lt__(self, other):
        # lexicographic ordering
        if DEBUG:
            k1_less = self.k1 < other.k1
            k2_less = np.isclose(self.k1, other.k1, atol=1e-5) and self.k2 < other.k2
            print(self.full_str() + " Vs " + other.full_str())
            if k1_less: print("K1 less")
            elif k2_less: 
                print("K2 less")
            else: print("Neither")

        return self.k1 < other.k1 or (
            np.isclose(self.k1, other.k1, atol=1e-5) and self.k2 < other.k2)

    def full_str(self):
        if np.isclose(self.k1, DstarLite.INF): k1 = "INF"
        else: k1 = "%.2f" % self.k1
        if np.isclose(self.k2, DstarLite.INF): k2 = "INF"
        else: k2 = "%.2f" % self.k2
        return "({},{}):[{},{}]".format(
            self.state[0], self.state[1], k1, k2)

    def __repr__(self):
        return "({},{})".format(
            self.state[0], self.state[1])

class DstarLite(object):
    INF = sys.float_info.max
    def __init__(self, graph, start, goal, four_connected=False):
        self.is_new = True
        self.graph = graph
        self.start = start
        self.goal = goal
        self.four_connected = four_connected
        self.km = 0  # changing start position and thus changing heuristic
        self.eps = 1.0
        self.G = np.ones(
            (self.graph.height, self.graph.width)) * DstarLite.INF
        self.V = np.ones(
            (self.graph.height, self.graph.width)) * DstarLite.INF
        self.updated_trans_costs = dict()

        # (cost, node), PQ uses first element for priority
        self.set_value(self.V, goal, 0)
        self.open_set = [self.create_node(goal)]
        self.successor = dict()
        self.successor[goal] = None
        self.path = []
        
    def compute_path_with_reuse(self, eps=1.0):
        # closed = set()
        # incons = []  # states with inconsistent values, but already expanded
        expansions = {}  # visualize order in which states are expanded
        # update whether start is inconsistent
        gstart = DstarLite.get_value(self.G, self.start)
        vstart = DstarLite.get_value(self.V, self.start)
        start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)
        if len(self.open_set) == 0:
            return expansions

        start_node = self.create_node(self.start)
        min_node = self.open_set[0]
        
        i = 0
        while len(self.open_set) > 0 and (
                (min_node < start_node) or start_inconsistent):
            # print("(Fstart, Fmin): (%s, %s)" % (start_node, min_node))
            # expand next node w/ lowest f-cost, add to closed
            print(self.open_set)
            cur_node = heapq.heappop(self.open_set)
            print("   Expanded %d : %s: %s" % (i, str(cur_node), str(list(self.graph.neighbors(cur_node.state)))))
            # print("Current: %s" % cur_node)
            cur_state = cur_node.state
            # closed.add(current)

            # visualize expansion process
            expansions[cur_state] = i
            i += 1

            g_cur = DstarLite.get_value(self.G, cur_state)
            v_cur = DstarLite.get_value(self.V, cur_state)
            if g_cur > v_cur:
                # g_cur = v_cur
                DstarLite.set_value(self.G, cur_state, v_cur)
                for next in self.graph.neighbors(cur_state): 
                    self.update_state(next)

            else:
                DstarLite.set_value(self.G, cur_state, DstarLite.INF)
                for next in self.graph.neighbors(cur_state): 
                    self.update_state(next)
                self.update_state(cur_state)  # Pred(s) U {s}

            # if reached start target state, update fstart value
            if DstarLite.state_equal(self.start, cur_state):
                start_node = self.create_node(self.start)

            # update whether start is inconsistent
            gstart = DstarLite.get_value(self.G, self.start)
            vstart = DstarLite.get_value(self.V, self.start)
            start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)

            # update fmin value
            if len(self.open_set) > 0:
                min_node = self.open_set[0]

        # add the leftover overconsistent states from incons
        # for anytime search
        # for v in incons: heapq.heappush(self.open_set, v)
        return expansions


    def remove_from_open(self, target_state):
        for i in range(len(self.open_set)):
            node = self.open_set[i]
            if DstarLite.state_equal(node.state, target_state):
                # set node to remove as last element, remove duplicate, reorder
                self.open_set[i] = self.open_set[-1]
                self.open_set.pop()  
                heapq.heapify(self.open_set)
                return

    def search(self, start):
        self.km += self.heuristic(self.start, start)
        self.start = start
        for next in self.graph.neighbors(self.start):
            # get transition cost, which may or may not change depending on env
            trans_cost, need_update = self.get_trans_cost(self.start, next, 
                observe_new_cost=True)

            # if trans-cost has changed and start was predecessor of next
            if need_update:
                self.update_state(self.start)
        
        # reverse start and goal so search backwards
        expansions = self.compute_path_with_reuse()
        print("Order of expansions:")
        viz.draw_grid(self.graph, width=3, number=expansions, start=start, 
            goal=self.goal)
        self.path = self.reconstruct_path()
        return self.path[1]  

    def get_min_g_val(self, cur):
        min_g = DstarLite.INF
        best_neighbor = None
        for next in self.graph.neighbors(cur):
            trans_cost, _ = self.get_trans_cost(cur, next)
            cost = (trans_cost + DstarLite.get_value(self.G, next))
            if cost < min_g or best_neighbor is None: 
                min_g = cost
                best_neighbor = next

        return min_g, best_neighbor


    def create_node(self, cur):
        k1, k2 = self.get_k(cur)
        return Node(k1, k2, cur)


    def get_k(self, cur):
        v = DstarLite.get_value(self.V, cur)
        g = DstarLite.get_value(self.G, cur)
        k2 = min(v, g)
        # k1 = f-value
        h = self.heuristic(cur, self.start)
        k1 = DstarLite.compute_f(g=k2, h=h, eps=self.eps)
        return (k1, k2)

    def update_state(self, cur_state):
        # if already in openset, need to remove since has outdated f-val
        self.remove_from_open(cur_state)

        # get updated g-value of current state
        if not DstarLite.state_equal(cur_state, self.goal):
            min_g, best_neighbor = self.get_min_g_val(cur_state)
            DstarLite.set_value(self.V, cur_state, min_g)
            self.successor[cur_state] = best_neighbor

        # if inconsistent, insert into open set
        v = DstarLite.get_value(self.V, cur_state)
        g = DstarLite.get_value(self.G, cur_state)
        if not np.isclose(v, g, atol=1e-5):
            heapq.heappush(self.open_set, self.create_node(cur_state))

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
        # path cost is symmetric between two states in gridworld, but not 
        # always true in real life (uphill v.s downhill)
        key = (cur, next)
        key2 = (next, cur)  
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
            self.updated_trans_costs[key] = cost
            print("%s -> %s: %d" % (str(cur), str(next), need_update))

        return cost, need_update


    def reconstruct_path(self):
        current = self.start
        path = []
        while not DstarLite.state_equal(current, self.goal):
            path.append(current)
            current = self.successor[current]
        path.append(self.goal)
        return path

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = a, b
        # Manhattan distance if four-connected
        if self.four_connected: return abs(x2-x1) + abs(y2-y1)
        # else Euclidean distance
        else: return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    @staticmethod
    def get_value(arr, node):
        x, y = node
        assert(0 <= x < arr.shape[1] and 0 <= y < arr.shape[0])
        assert(isinstance(x, int) and isinstance(y, int))
        return arr[y,x]

    @staticmethod
    def set_value(arr, node, val):
        x, y = node
        assert(0 <= x < arr.shape[1] and 0 <= y < arr.shape[0])
        assert(isinstance(x, int) and isinstance(y, int))
        arr[y,x] = val

    @staticmethod
    def compute_f(g, h, eps):
        # just to be explicit
        return g + eps*h

    @staticmethod
    def state_equal(n1, n2):
        x1, y1 = n1
        x2, y2 = n2
        return np.isclose(x1, x2, atol=1e-5) and np.isclose(y1, y2, atol=1e-5)


def test():
    start = (3, 0)
    goal = (6, 6)  # (x, y)
    width = 3
    four_connected = False
    # In diagram_simple test, you will notice robot constantly attempt to move 
    # through (0,1) as that is the shortest way to goal
    # it doesn't associate a state with transition cost, but rather a specific 
    # (cur, next) pair which is why it keeps attempting these until it's clear 
    # going through (0, 1) won't work
    # viz.diagram_simple.four_connected = four_connected
    # planner = DstarLite(viz.diagram_simple, start, goal, four_connected)

    viz.diagram2.four_connected = four_connected
    planner = DstarLite(viz.diagram2, start, goal, four_connected)
    while not DstarLite.state_equal(start, goal):
        next = planner.search(start)

        # visualize stuff
        print()
        viz.draw_grid(planner.graph, width, 
            path=planner.path, start=start, goal=goal)

        start = next
        print("New Start: %s" % str(start))
    # viz.draw_grid(viz.diagram4, width, number=cost_so_far, start=start, goal=goal)
    # print()
    # viz.draw_grid(viz.diagram4, width, 
    #     path=reconstruct_path(came_from, start=start, goal=goal))


if __name__=="__main__":
    test()