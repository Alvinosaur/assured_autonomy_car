import heapq
import numpy as np
import sys
import copy
import math
import matplotlib.pyplot as plt

from dstar_lite import Node, DstarLite

DEBUG = False
RAD_TO_DEG = 180.0 / math.pi

"""
Confusions:
1. how to give transition between negative and positive velocity? Currently have vel=0 option, but there  is no trajectory associated with this. Maybe just have x,y remain the same at that point, and low-level controller will naturally slow down and stay at this point more

2. currently not replanning at each step in a lattice path, just replanning at the end of the lattice path. Problems include 
    1: What if overshoot goal? That means getting the neighbor of a given state cannot just be the end of that lattice path, but need to look at all intermediate states. 
    
    2. What defines "reaching goal"? Just position distance, or also velocity and heading? If we reached goal in position but not in velocity (ie: moving too fast), then just don't treat as goal, and need to consider adjacent neighbors with adjacent velocity profiles, then planner will hopefully expand a previous state and move to goal with slower velocity.
    
    3. Continuing off #2, should velocity be included in heuristic? If goal vel =0, wouldn't there always be a bias of driving with velocity near 0? Would need to ensure reaching goal in minimal time is higher priority than moving at goal velocity. 

    3. without replanning at every intermediate state along a lattice path, are losing a lot of possible solutions. In extreme case, if lattice paths are too large, may be impossible to exactly land on goal. 

3. When we observe new terrain around robot, how to figure out which paths lie in this region? Can't just check which start/end nodes lie in region because a trajectory may cross over this region but start and stop outside it... Need an  efficient way to query all connections crossing over a given position (which will include multiple states w/ different thetas and velocities).

"""


class LatticeNode(Node):
    """Same implementation as normal Node except with extended state:
    state = [x,y,theta,vel]

    Args:
        Node ([type]): [description]
    """

    def __init__(self, k1, k2, state):
        super().__init__(k1, k2, state)
        assert(len(state) == 4)  # 4 states for [x,y,theta,vel]

    def full_str(self):
        if np.isclose(self.k1, DstarLite.INF):
            k1 = "INF"
        else:
            k1 = "%.2f" % self.k1
        if np.isclose(self.k2, DstarLite.INF):
            k2 = "INF"
        else:
            k2 = "%.2f" % self.k2
        return "({},{},{},{}):[{},{}]".format(
            self.state[0], self.state[1], self.state[2], self.state[3], k1, k2)

    def __repr__(self):
        return "({},{},{},{})".format(
            self.state[0], self.state[1], self.state[2], self.state[3])


class Graph(object):
    def __init__(self, map, minx, miny, dx, dy, thetas,
                 state_to_key_fn, base_prims,
                 base_prims_trajs, viz=False):
        self.map = map
        assert(isinstance(map, np.ndarray))
        self.height, self.width = map.shape  # x, y
        self.minx = minx
        self.miny = miny
        self.dx = dx
        self.dy = dy
        self.thetas = thetas
        assert(len(thetas) > 1)
        self.state_to_key_fn = state_to_key_fn
        self.mprim_actions = base_prims
        self.mprim_trajs = base_prims_trajs
        self.all_disc_traj = []
        assert(len(self.mprim_actions) == len(self.mprim_trajs))

        # compute discretized mprim trajectories
        # TODO: more accurate way is to compute for every x,y in trajectory
        # the overlapping cells of car(need width,length of car)
        # but for now just use the x,y states only

        # T x 3 X N for T timesteps and N primitives
        # organized as [x,y,theta]
        # precompute all possible motion prim trajectories from all headings
        assert (len(self.mprim_trajs.shape) == 3)
        num_prims = self.mprim_trajs.shape[2]
        for theta_offset in thetas:
            new_prims = np.copy(self.mprim_trajs)
            new_prims[:, 2, :] += theta_offset
            rot_mat = np.array([
                [math.cos(theta_offset), -math.sin(theta_offset)],
                [math.sin(theta_offset), math.cos(theta_offset)]])
            for prim_i in range(num_prims):
                yx_coords = np.fliplr(new_prims[:, 0:2, prim_i])
                new_prims[:, 0:2, prim_i] = np.fliplr(yx_coords @ rot_mat)

                if viz:
                    theta_deg = theta_offset * RAD_TO_DEG
                    plt.plot(new_prims[:, 0, prim_i],
                             new_prims[:, 1, prim_i])
            if viz:
                plt.title("Mprims for theta: %.2fº" % theta_deg)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
                plt.close()
            self.all_traj.append(new_prims)
            self.all_disc_traj.append(
                self.state_to_key_fn(new_prims, is_traj=True))

    def neighbors(self, state):
        x, y, theta, v = state
        _, _, theta_i, _ = self.state_to_key_fn(state)
        assert (0 <= theta_i < len(self.thetas))
        base_prims = self.all_traj[theta_i]

        return base_prims +
        # need to find which discrete states the trajectory covers
        # could be pre-computed when given the continuous-space trajectory?

    def cost(self, cur, next, observe_new_cost):
        return 0


class LatticeDstarLite(DstarLite):
    def __init__(self, graph, velocities, thetas, start, goal):
        # assume uniform discretization of velocity and thetas
        self.velocities = velocities
        self.vel_min_max = [min(self.velocities), max(self.velocities)]
        if len(self.velocities) < 1:
            self.dv = 0
        else:
            self.dv = abs(velocities[0] - velocities[1])
        assert(len(thetas) > 0)
        self.thetas = thetas
        self.theta_min_max = [min(self.thetas), max(self.thetas)]

        # thetas should be [0, dtheta, ... 2pi - dtheta]
        assert (np.isclose(self.theta_min_max[0], 0, rtol=1e-5))
        self.dtheta = abs(thetas[0] - thetas[1])
        assert (np.isclose(
            self.theta_min_max[1], math.pi - self.dtheta, rtol=1e-5))
        self.PI_i = len(thetas)

        # half of bin is the limit where value considered same bin
        self.min_cont = np.array(
            [self.graph.minx, self.graph.miny,
             self.theta_min_max[0], self.vel_min_max[0]])
        self.cont_to_disc = np.array(
            [graph.dx, graph.dy, self.dtheta, self.dv])
        self.tolerance_vec = self.cont_to_disc / 2.0

        # in case goal isn't reachable from start, mark closest to start
        start = np.array(start)
        goal = np.array(goal)
        self.closest_to_start = goal
        self.best_dist = np.linalg.norm(goal[:2] - start[:2])

        # only difference is that G and V value tables are 4 dimensional
        super().__init__(graph=graph, four_connected=False)
        self.state_space_dim = (self.graph.width, self.graph.height,
                                len(self.velocities), len(self.thetas))
        self.G = np.ones(shape=self.state_space_dim) * self.INF
        self.V = np.copy(self.G)

    def compute_path_with_reuse(self, eps=1.0):
        # closed = set()
        # incons = []  # states with inconsistent values, but already expanded
        expansions = {}  # visualize order in which states are expanded
        # update whether start is inconsistent
        gstart = self.get_value(self.G, self.start)
        vstart = self.get_value(self.V, self.start)
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
            print("   Expanded %d : %s: %s" % (i, str(cur_node),
                                               str(list(self.graph.neighbors(cur_node.state)))))
            # print("Current: %s" % cur_node)
            cur_state = cur_node.state
            # closed.add(current)

            # visualize expansion process
            expansions[cur_state] = i
            i += 1

            g_cur = self.get_value(self.G, cur_state)
            v_cur = self.get_value(self.V, cur_state)
            if g_cur > v_cur:
                # g_cur = v_cur
                self.set_value(self.G, cur_state, v_cur)
                for next in self.graph.neighbors(cur_state):
                    self.update_state(next)

            else:
                self.set_value(self.G, cur_state, self.INF)
                for next in self.graph.neighbors(cur_state):
                    self.update_state(next)
                self.update_state(cur_state)  # Pred(s) U {s}

            # if reached start target state, update fstart value
            if self.state_equal(self.start, cur_state):
                start_node = self.create_node(self.start)

            # update whether start is inconsistent
            gstart = self.get_value(self.G, self.start)
            vstart = self.get_value(self.V, self.start)
            start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)

            # update fmin value
            if len(self.open_set) > 0:
                min_node = self.open_set[0]

        # add the leftover overconsistent states from incons
        # for anytime search
        # for v in incons: heapq.heappush(self.open_set, v)
        return expansions

    def update_closest_to_start(self, cur_state):
        dist = np.linalg.norm(cur_state[:2] - self.start[:2])
        if dist < self.best_dist:
            self.best_dist = dist
            self.closest_to_start = np.copy(cur_state)

    def create_node(self, cur):
        k1, k2 = self.get_k(cur)
        return LatticeNode(k1, k2, cur)

    def state_to_key(self, state, is_traj=False):
        if is_traj:
            assert (isinstance(state, np.ndarray))
            return np.round(
                (state - self.min_cont) / self.cont_to_disc).astype(int)
        else:
            state_key = np.round(
                (np.array(state) - self.min_cont) / self.cont_to_disc).astype(int)
            xi, yi, thetai, vi = state_key
            # handle angle wraparound after rounding to nearest int in case
            # floating error causes modulo operation to mess up
            thetai = thetai % self.PI_i
            assert (self.is_valid_key(state_key=(xi, yi, thetai, vi)))
            return (xi, yi, thetai, vi)

    def is_valid_key(self, state_key):
        xi, yi, thetai, vi = state_key
        return ((0 <= xi < self.graph.width) and
                (0 <= yi < self.graph.height) and
                (0 <= thetai < self.PI_i) and
                (0 <= vi < len(self.velocities)))

    def key_to_state(self, key):
        return np.array(key) * self.cont_to_disc

    def update_state(self, cur_state):
        # if already in openset, need to remove since has outdated f-val
        self.remove_from_open(cur_state)

        # get updated g-value of current state
        if not self.state_equal(cur_state, self.goal):
            min_g, best_neighbor = self.get_min_g_val(cur_state)
            self.set_value(self.V, cur_state, min_g)
            self.successor[self.state_to_key(cur_state)] = best_neighbor

        # if inconsistent, insert into open set
        v = self.get_value(self.V, cur_state)
        g = self.get_value(self.G, cur_state)
        if not np.isclose(v, g, atol=1e-5):
            heapq.heappush(self.open_set, self.create_node(cur_state))

    def get_and_update_trans_cost(self, cur, next, observe_new_cost=False):
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
        (x1, y1) = cur[:2]
        (x2, y2) = next[:2]

        # if already found true cost, simply return it, need_update=False
        # path cost is symmetric between two states in gridworld, but not
        # always true in real life (uphill v.s downhill)
        cur_key, next_key = self.state_to_key(cur), self.state_to_key(next)
        key = cur_key + next_key
        key2 = next_key + cur_key
        if key in self.updated_trans_costs:
            return self.updated_trans_costs[key], False
        elif key2 in self.updated_trans_costs:
            return self.updated_trans_costs[key2], False

        cost, need_update = self.graph.cost(cur, next, observe_new_cost)
        if observe_new_cost:
            # updated_trans_costs only stores ground-truth costs, so
            # only store new cost if able to observe new, true costs
            # TODO: ideally, we shouldn't hardcode updated transition cost
            # and keep reusing this as robot may continue to improve its
            # cost observations, and thus we should replan if cost discrepancy
            # high enough
            self.updated_trans_costs[key] = cost
            print("%s -> %s: %d" % (str(cur), str(next), need_update))

        return cost, need_update

    def heuristic(self, a, b):
        # TODO: Euclidean distance grossly underestimates true
        # cost for dynamically-constrained vehicles, should find
        # a better heuristic
        (x1, y1), (x2, y2) = a, b
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    def state_equal(self, n1, n2):
        # let state_to_key handle issues of angle wraparound
        n1_key = np.array(self.state_to_key(n1))
        n2_key = np.array(self.state_to_key(n2))
        return abs(n1_key - n2_key) < self.tolerance_vec

    def get_value(self, arr, node):
        xi, yi, thetai, vi = self.state_to_key(node)
        return arr[xi, yi, thetai, vi]

    def set_value(self, arr, node, val):
        xi, yi, thetai, vi = self.state_to_key(node)
        arr[xi, yi, thetai, vi] = val

    @staticmethod
    def compute_f(g, h, eps):
        # just to be explicit
        return g + eps*h


def test_state_equal():
    """
    test upper and lower bounds of half bins
    test that out-of-bound assertions are called
    """
    # define position space
    dy, dx = 0.1, 0.1
    miny, minx = -2, -2
    maxy, maxx = 8, 8
    Y = int(round((maxy - miny + dy) / dy))  # [min, max] inclusive
    X = int(round((maxx - minx + dx) / dx))
    map = np.zeros((Y, X))

    # define velocity space
    velocities = np.linspace(start=-2, stop=2, num=5)
    dv = 1

    # define heading space
    thetas = np.linspace(start=0, stop=135, num=4)
    # NOTE: Actual thetas are in radians! here just shown in degrees for clarity
    assert (np.allclose(thetas, [0, 45, 90, 135]))
    thetas = thetas / RAD_TO_DEG  # convert to radians
    dtheta = 45 / RAD_TO_DEG

    # load in pre-generated motion primitives
    mprim_file = "search_planning_algos/motion_primitives.npz"
    data = np.load(mprim_file)
    mprims = data["prim_trajs"]

    # arbitrary
    start = [0, 0, 0, 0]
    goal = [5, 5, 0, 0]
    graph = Graph(map=map, minx=minx, miny=miny, dx=dx,
                  dy=dy, thetas=thetas, base_prims=mprims, viz=False)
    planner = LatticeDstarLite(graph, velocities, thetas, start, goal)

    # test that upper and lower bounds map to same key for different states
    test_states = [
        [minx, miny, thetas[0], velocities[0]],
        [maxx, maxy, thetas[-1], velocities[-1]]
    ]
    for state in test_states:
        s1 = np.array(state)
        s1_key = planner.state_to_key(s1)
        cont_to_disc = np.array([dx, dy, dtheta, dv])
        # subtract 1e-10 to break tie (ie: 5.5 -> 6, but 5.499 -> 5) for python2.7
        # NOTE: using hard-coded 1e-10 to break ties may cause bug if level
        # of discretization is finer than 1e-10
        upper_s1 = s1 + (cont_to_disc / 2.0) - 1e-10
        lower_s1 = s1 - (cont_to_disc / 2.0) + 1e-10
        assert (s1_key == planner.state_to_key(upper_s1))
        assert(s1_key == planner.state_to_key(lower_s1))

    # make it explicit how rounding is done, very hard-coded
    s1 = np.array([8, 8, 135 / RAD_TO_DEG, 2])
    # with python 3, 0.5 rounds down to 0, so make tests work for both versions
    upper_s1 = s1 + np.array([0.0499, 0.0499, 22.499 / RAD_TO_DEG, 0.499])
    assert (planner.state_to_key(s1) == planner.state_to_key(upper_s1))
    past_s1 = s1 + np.array([0.051, 0.051, 22.51 / RAD_TO_DEG, 0.51])
    s1_key = np.array(planner.state_to_key(s1))
    try:
        past_s1_key = np.array(planner.state_to_key(past_s1))
        print("Values should be out-of-bounds and raise assertion!")
        assert(False)
    except:
        pass
        # # past_s1_key has indexes offset by +1 except for last entry
        # assert (np.allclose(s1_key[0:3] + 1, past_s1_key[0:3]))
        # # last entry of past_s1_key wraps around back to theta=0, or idx=0
        # assert(np.isclose(past_s1_key[3], 0))


# def test_graph_cost():
#     # test whether graph correctly determines intermediate
#     # states from two states
#     # test whether total path cost between two states makes sense


if __name__ == "__main__":
    test_state_equal()
