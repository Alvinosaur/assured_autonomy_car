import heapq
import numpy as np
import sys
import copy
import math
import matplotlib.pyplot as plt
import time
import kdtree
import dubins
from astarlib import astarlib

from car_dynamics import Action, Car
from lattice_graph import Graph

DEBUG = False
RAD_TO_DEG = 180.0 / math.pi
TWO_PI = 2 * math.pi

"""
Confusions:
1. how to give transition between negative and positive velocity? Currently have vel=0 option, but there  is no trajectory associated with this. Maybe just have x,y remain the same at that point, and low-level controller will naturally slow down and stay at this point more

2. currently not replanning at each step in a lattice path, just replanning at the end of the lattice path. Problems include
    1: What if overshoot goal? That means getting the neighbor of a given state cannot just be the end of that lattice path, but need to look at all intermediate states.

    2. What defines "reaching goal"? Just position distance, or also velocity and heading? If we reached goal in position but not in velocity (ie: moving too fast), then just don't treat as goal, and need to consider adjacent neighbors with adjacent velocity profiles, then planner will hopefully expand a previous state and move to goal with slower velocity.

    3. Continuing off #2, should velocity be included in heuristic? If goal vel =0, wouldn't there always be a bias of driving with velocity near 0? Would need to ensure reaching goal in minimal time is higher priority than moving at goal velocity.

    3. without replanning at every intermediate state along a lattice path, are losing a lot of possible solutions. In extreme case, if lattice paths are too large, may be impossible to exactly land on goal.

3. When we observe new terrain around robot, how to figure out which paths lie in this region? Can't just check which start/end nodes lie in region because a trajectory may cross over this region but start and stop outside it... Need an  efficient way to query all connections crossing over a given position (which will include multiple states w/ different thetas and velocities).

NOTE: It looks like goal should face up, but theta is chosen in image-frame as is the y-coordinates, so -90 faces upwards on our screen and +90 faces down...
If we choose goal with theta = +90, the first set of expansions we see seem to all face upwards, but actually +90 corresponds  to facing  down. The reason they "look like" they're all facing up is that we're visualizing  predecessors of the goal state, or all the possible trajectories that could lead directly to the goal.

Differences from original dstar-lite:
- for original, I could use the same "get-neighbors" function to get both predecessors and successors. Here, however, predecessors != sucessors since theta(heading) controls which direction car is moving. We account for this by simply flipping the current heading by 180 degrees and running rollout generation to get predecessors. NOTE: Especially for 3D terrain, moving forwards and backwards over the same path is not the same (ex: fall over cliff v.s climb wall). But temporarily, this approach works with our cost functions and heuristic. If we want predecessor of current state facing forwards, need to flip theta by 180deg so state faces backwards. The our rollouts face backwards, but their orientations are now wrong (since also facing backwards when they should face forwards), so need to flip all their orientations back.

- Since non-holonomic constraint of ackermann steering, we use a different heuristic:
-  Euclidean distance underestimates path costs by violating kinematic constraints of robots. This misleads the search and leads to local minima where robot gets to the desired location, but not with the desired heading.

For testing with a map generated from image and picking points using matplotlib GUI, (x,y) is valid, but our orientation is wrong. Moving up screen is -90, moving down is +90deg.

On 7/19, added time to state so plan not only finds shortest-path, but also fastest path since two states with same position but diff velocities are always expanded together.

CrAzY BuG: tried using actual, absolute time-value as part of the transition cost... transition cost is defined as cost to transition from one state to another, so time should be time taken to get from state 1 to state 2, not from very beginning to state 2. This bug caused the planner to keep choosing paths with small time values.

TODOS:
- get_and_update_trans_cost()... bullet point #3 above
- use rollout generator with dynamics rather than current approach of kinematics
- clean up LatticeDstarLite.__init__()

Implementation:
- trajectory rollouts created using basic bicycle kinematic model
- collision-checking uses wheel radius and checks if change in elevation > radius
- Currently online observation windows are squares with width and height rather than circle with radius

Tests:
Graph.update_map:
- verify window at limits of map
- verify need_update calculated correctly

Graph.calc_cost_and_validate_traj:
- verify that trajectory gets truncated when becomes invalid and no costs are None
- verify distance-traveled is 3D euclidean

Graph.generate_trajectories_and_costs:
- verify traj_sample_distances contains 2D distance traveled
- visualize to double-check that can't move both forward and backwards from a given state, only one or the other
"""


class LatticeNode():
    """Same implementation as normal Node except with extended state:
    state = [x,y,theta,vel]

    Args:
        Node ([type]): [description]
    """

    def __init__(self, k1, k2, state):
        assert (len(state) == 5)  # 4 states for [x,y,theta,vel,t]
        self.k1 = k1
        self.k2 = k2
        self.state = state

    def __lt__(self, other):
        # compare k1 first and k2 after if tie
        return self.k1 < other.k1 or (
            np.isclose(self.k1, other.k1, atol=1e-5) and self.k2 < other.k2)

    def __repr__(self):
        return "%.2f,%.2f,%.2f,%.2f,%.2f, k1, k2: %.2f, %.2f" % (
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.k1, self.k2)


class LatticeDstarLite(object):
    INF = 1e10
    MAX_THETA = 2 * math.pi

    def __init__(self, graph: Graph, car: Car, min_state, dstate, velocities, steer_angles, thetas, T, goal_thresh=None, eps=1.0, viz=True, reconstruct_full=True):
        self.update_state_time = 0
        self.min_state = min_state
        self.dstate = dstate
        self.velocities = velocities
        self.thetas = thetas
        self.steer_angles = steer_angles
        self.car = car
        self.min_turn_rad = car.L / math.tan(car.max_steer)
        self.reconstruct_full = reconstruct_full
        self.viz = viz
        if viz:
            f = plt.figure()
            self.ax = f.add_subplot(111)
            self.ax.imshow(graph.map)
            plt.ion()
            plt.show()

        # define action space
        self.graph = graph
        self.obstacle_heuristic = astarlib.aStar(self.graph.obstacle_map)
        self.actions = [Action(v=v, steer=steer)
                        for steer in steer_angles for v in velocities]
        self.dt = self.get_time(state=dstate)
        self.T = T
        self.N = int(self.T / self.dt)
        self.precompute_all_primitives()

        if goal_thresh is None:
            self.goal_thresh = 3 * np.linalg.norm(self.dstate[:2])
        else:
            self.goal_thresh = goal_thresh
        self.heading_thresh = self.get_theta(state=dstate)

        self.is_new = True
        self.shifted_start = False
        self.eps = eps

        # find nearest neighbors by position
        self.kdtree = kdtree.create(dimensions=2)
        self.dist_thresh = (self.dstate[0]**2 + self.dstate[1]**2)
        self.neighbor_count_thresh = 7

        # values that can change if new goal is set
        self.km = 0  # changing start position and thus changing heuristic

        # populated during call to self.search()
        self.successor = dict()
        self.path = []
        self.policy = []
        self.open = []
        self.open_set = set()
        self.start = self.goal = None
        self.start_key = self.goal_key = None
        self.path_i = 0

        # G and V values explained in dstar_lite.py
        self.G = dict()
        self.V = dict()

        # benchmark debugging
        self.update_state_time = 0
        self.remove_from_open_time = 0

    def precompute_all_primitives(self):
        # not a fully-defined state, but used as origin
        x0, y0, theta0 = 0, 0, 0
        state0 = np.array([x0, y0, theta0])

        # N x 5 X M for N timesteps and M primitives
        self.base_trajs = np.zeros(
            shape=(self.N, len(self.dstate), len(self.actions)))
        for ai, action in enumerate(self.actions):
            self.base_trajs[:, :, ai] = self.car.rollout(
                state=state0, action=action, dt=self.dt, T=self.T, t0=0)

        self.graph.generate_trajectories_and_costs(
            mprim_actions=self.actions, base_trajs=self.base_trajs, state0=state0)

    def euc_dist(self, s1, s2):
        return np.linalg.norm(s1[:2] - s2[:2])

    def set_new_start(self, start):
        self.shifted_start = False  # reset
        self.start_key = self.state_to_key(start)
        assert (self.is_valid_key(self.start_key))
        if self.start is not None:
            # not using heuristic here since dubins path distance between two slightly shifted, nearby states can be huge
            self.km += self.euc_dist(self.start, start)
        self.start = np.copy(start)

    def set_new_goal(self, goal):
        self.goal = np.copy(goal)
        self.goal_key = self.state_to_key(goal)
        assert(self.is_valid_key(self.goal_key))

        self.V = dict()
        self.G = dict()
        self.set_value(val_map=self.V, state_key=self.goal_key, val=0)

        # unsorted set version of "open" which is actually a heap, but is termed "open" by convention
        self.open = [self.create_node(goal)]
        self.open_set = {self.goal_key}

        self.successor = dict()
        self.successor[self.goal_key] = None
        self.path = []
        self.policy = []
        self.expand_order = []
        self.km = 0
        self.path_i = 0

    def clean_up_open_(self):
        """lazy removal policy where open_set the true depiction
        of which states are in open set
        This function cleans up self.open to match states in open_set. This
        is implemented as such to allow for O(1) state-lookup while still
        taking advantage of heaps for min-cost state expansion
        """
        found_stale = True
        while found_stale and len(self.open) > 0:
            min_node = self.open[0]
            if self.state_to_key(min_node.state) not in self.open_set:
                heapq.heappop(self.open)
            else:
                found_stale = False

    def add_to_open(self, node: LatticeNode):
        state_key = self.state_to_key(node.state)
        heapq.heappush(self.open, node)
        self.open_set.add(state_key)

    def pop_from_open(self):
        self.clean_up_open_()
        try:
            cur_node = heapq.heappop(self.open)
            cur_key = self.state_to_key(cur_node.state)
            self.open_set.remove(cur_key)
            return cur_node, cur_key
        except IndexError:
            print("Trying to pop from empty open!")
            return None, None

    def peek_min_open(self):
        self.clean_up_open_()
        if len(self.open) > 0:
            return self.open[0]
        else:
            print("Empty open, cannot peek!")
            return None

    def remove_from_open(self, state_key):
        """Removes a state(with specified key) if it lies in open_set.
        Does NOT remove from self.open and leaves this to other helper functions
        so self.open and self.open_set will not be identical all the time.

        Args:
            state_key ([type]): [description]
        """
        if state_key in self.open_set:
            self.open_set.remove(state_key)

    def add_to_kdtree(self, state):
        self.kdtree.add(state[:2])

    def compute_path_with_reuse(self):
        # update whether start is inconsistent
        gstart = self.get_value(self.G, state_key=self.start_key)
        vstart = self.get_value(self.V, state_key=self.start_key)
        start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)

        if len(self.open_set) == 0:
            return
        start_node = self.create_node(self.start)
        min_node = self.peek_min_open()

        while len(self.open_set) > 0 and (
                (min_node < start_node) or start_inconsistent):
            # print("(Fstart, Fmin): (%s, %s)" % (start_node, min_node))
            # expand next node w/ lowest f-cost, add to closed
            # print(self.open)

            cur_node, cur_key = self.pop_from_open()
            cur_state = cur_node.state
            print("Expanded: %s, f=%.2f" %
                  (self.state_to_str(cur_state), cur_node.k1))

            # track order of states expanded
            self.expand_order.append(cur_state)

            # update current state's G value with V
            cur_G = self.get_value(self.G, state_key=cur_key)
            cur_V = self.get_value(self.V, state_key=cur_key)
            if cur_G > cur_V:
                cur_G = cur_V
                self.set_value(self.G, state_key=cur_key, val=cur_G)
            else:
                cur_G = self.INF
                self.set_value(self.G, state_key=cur_key, val=cur_G)
                self.update_state(cur_state)  # Pred(s) U {s}

            # if reached start target state, update fstart value
            # implement goal threshold by applying threshold to start since backwards search. If any state is within threshold of start, just pick this state as new start.
            if self.reached_target(cur_state, self.start):
                # time of course won't be the same
                if not self.state_equal(self.start, cur_state, ignore_time=True):
                    self.set_new_start(cur_state)
                    self.shifted_start = True
                    start_node = self.create_node(self.start)
                    # return entire path, not just path[1:]

            # get neighbor states update their values if necessary
            all_trajs, dist_costs, actions = self.graph.neighbors(
                cur_state, predecessor=True)
            for ti, traj in enumerate(all_trajs):
                # iterate through trajectory and add states to open set
                # print()
                # print("Action: %s" % str(actions[ti]))
                for si in range(traj.shape[0]):
                    next_state = traj[si, :]
                    new_G = (dist_costs[ti][si] + cur_G)
                    self.update_state(
                        next_state, predecessor_key=cur_key, new_G=new_G, action=actions[ti], t=si * self.dt)

                if self.viz:
                    dx, dy, _, _, _ = self.dstate
                    x_data = traj[:, 0] / dx
                    y_data = traj[:, 1] / dy
                    self.ax.scatter(x_data, y_data, s=2)
                    plt.draw()

            if self.viz:
                plt.pause(0.5)

            # TODO: if reached start target state, update fstart value? Original dstar_lite doesn't do anything meaningful here

            # update whether start is inconsistent
            gstart = self.get_value(self.G, state_key=self.start_key)
            vstart = self.get_value(self.V, state_key=self.start_key)
            start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)

            # update fmin value
            min_node = self.peek_min_open()

        # add the leftover overconsistent states from incons
        # for anytime search
        # for v in incons: heapq.heappush(self.open, v)

    def search(self, start, goal, obs_window, window_bounds):
        self.set_new_start(start)

        # time irrelevant for goal so ignore
        if self.goal is None or not self.state_equal(self.goal, goal, ignore_time=True):
            self.set_new_goal(goal)

        # in case goal isn't reachable from start, mark closest to start
        # TODO: unused at the moment
        self.closest_to_start = self.goal
        self.best_dist = np.linalg.norm(self.goal[:2] - self.start[:2])

        # simulate act of observing new terrain near robot
        need_update = self.graph.update_map(
            obs_window=obs_window, xbounds=window_bounds[0], ybounds=window_bounds[1])

        if self.viz:
            self.ax.clear()
            self.ax.imshow(self.graph.map)

        if need_update and len(self.open) > 0:
            # point of dstar lite is to not update every state in open-set
            # but only those that are relevant, so  only update start and let this propagate
            self.path_i = 0
            self.update_state(self.start)
            self.obstacle_heuristic = astarlib.aStar(self.graph.obstacle_map)

        if need_update or self.path == []:
            # reverse start and goal so search backwards
            self.compute_path_with_reuse()
            # print("Order of expansions:")
            # viz.draw_grid(self.graph, width=3, number=expansions, start=start,
            #               goal=self.goal)
            self.path, self.policy = self.reconstruct_path()
            return self.path, self.policy
        else:
            self.path_i += 1
            return self.path[self.path_i:], self.policy[self.path_i:]

    def update_closest_to_start(self, cur_state):
        # TODO: Currently unused
        dist = np.linalg.norm(cur_state[:2] - self.start[:2])
        if dist < self.best_dist:
            self.best_dist = dist
            self.closest_to_start = np.copy(cur_state)

    def get_min_g_val(self, cur):
        """Find best "successor"(predecessor in search process) from current state. Since we are doing backward search, all "successor" states are actually predecessors in the search process (ie: they were expanded before this current state in temporal order). In this case, when looking for neighbor states, these states' time values need to be subtracted in sequene rather than added.

        neighbors: (start) 5 <- 4 <- 3 <- 2, 1 <- 0 (goal)
        original neighbors: (start) 4 -> 5 -> 6 -> 7 -> 8 (goal)
            NOTE: notice above timesteps for start's "successors" are wrong wrt search process but correct in actual execution

        new neighbors offset: (start) 4 -> 3 -> 2 -> 1 -> 0 (goal)

        Args:
            cur ([type]): [description]

        Returns:
            [type]: [description]
        """
        min_g = self.INF
        best_neighbor_info = None
        all_trajs, trans_cost, actions = self.graph.neighbors(
            cur, backwards=True)
        for ti, traj in enumerate(all_trajs):
            # iterate through trajectory and add states to open set
            for si in range(traj.shape[0]):
                next_state = traj[si, :]
                assert(np.isclose(self.get_vel(
                    state=next_state), self.actions[ti].v))
                cost = (trans_cost[ti][si] +
                        self.get_value(self.G, state=next_state))
                if cost < min_g or best_neighbor_info is None:
                    min_g = cost
                    best_neighbor_info = (
                        next_state, actions[ti], si * self.dt)

        return min_g, best_neighbor_info

    def create_node(self, cur):
        k1, k2 = self.get_k(cur, target=self.start)
        return LatticeNode(k1, k2, cur)

    def get_k(self, cur, target):
        if np.allclose(cur, [63.11, 40.15, 0.00, 20.00, 4.30]):
            print("Hi")
        v = self.get_value(self.V, state=cur)
        g = self.get_value(self.G, state=cur)
        k2 = min(v, g)
        # k1 = f-value
        h = self.heuristic(cur, target)
        # km here accounts for issue where the moving start is our "goal" with
        # backward search, so the heuristics always change, but can just add
        # constant
        k1 = self.compute_f(g=k2, h=h + self.km, eps=self.eps, )
        # print("%s: g(%.2f) + eps*h(%.2f) = (%.2f):" % (
        #     self.state_to_str(cur), k2, k1 - k2, k1))
        return (k1, k2)

    def state_to_key(self, state):
        """Discretizes a state with self.dstate. Takes care of discretized
        theta wraparound. Returns hashable tuple.

        Args:
            state ([type]): [description]

        Returns:
            [type]: [description]
        """
        state_key = np.round(
            (np.array(state) - self.min_state) / self.dstate).astype(int)

        # wrap around theta back to [0, 2pi]
        state_key[self.get_theta(index=True)] %= len(self.thetas)
        return tuple(state_key)

    def is_valid_key(self, state_key):
        xi, yi, thetai, vi, ti = state_key
        return ((0 <= xi < self.graph.width) and
                (0 <= yi < self.graph.height) and
                (0 <= thetai < len(self.thetas)) and
                (0 <= vi < len(self.velocities)) and
                0 <= ti)

    def key_to_state(self, key):
        return (np.array(key) * self.dstate) + self.min_state

    def is_duplicate(self, state):
        nn = self.kdtree.search_nn_dist(
            point=state[:2], distance=self.dist_thresh)
        # kdtree.visualize(self.kdtree)
        return False
        return len(nn) >= self.neighbor_count_thresh

    def update_state(self, cur_state, action=None, t=None, predecessor_key=None, new_G=None):
        start_time = time.time()
        # if already in openset, need to remove since has outdated f-val
        cur_key = self.state_to_key(cur_state)
        self.remove_from_open(cur_key)

        # get updated v, g-value of current state
        cur_V = self.get_value(self.V, state_key=cur_key)
        cur_G = self.get_value(self.G, state_key=cur_key)

        # using direct comparison for increased performace
        if cur_key != self.goal_key:
            if predecessor_key is not None:
                if cur_V > new_G:
                    cur_V = new_G
                    self.set_value(self.V, state_key=cur_key, val=cur_V)
                    self.successor[cur_key] = (predecessor_key, action, t)
            else:
                min_g, best_neighbor_info = self.get_min_g_val(cur_state)
                self.set_value(self.V, state_key=cur_key, val=min_g)
                self.successor[cur_key] = best_neighbor_info

        # if inconsistent, insert into open set

        if not np.isclose(cur_V, cur_G, atol=1e-5) and (
                not self.is_duplicate(cur_state)):
            self.add_to_open(self.create_node(cur_state))
            self.add_to_kdtree(cur_state)

        end_time = time.time()
        self.update_state_time += (end_time - start_time)

    def reconstruct_path(self):
        """Generates full state path and policy (actions and their durations)
        after solving for motion plan. By default, returns the exact state path found, which can skip over intermediary states. This skipping behavior
        can be avoided by specifying reconstruct_full=True at init, which will have every state along path returned and action durations all of value dt. To return full path, we use car controller to generate all the intermediary steps.

        Returns:
            tuple(state path, policy): 
        """
        cur_key = self.start_key
        path = []
        policy = []
        if self.shifted_start:
            # add start if this start isn't exactly original start
            path.append(self.start)

        while cur_key != self.goal_key:
            (next_key, action, t) = self.successor[cur_key]
            cur_state = self.key_to_state(cur_key)
            next_state = self.key_to_state(next_key)

            # since plan was searched backwards, need to reverse steering
            action.steer *= -1
            if self.reconstruct_full:
                rollout = self.car.rollout(
                    state=cur_state, action=action, dt=self.dt, T=t, t0=0)
                N = int(t / self.dt)
                print("Rollout  generating for N= %d" % N)
                for i in range(N):
                    path.append(rollout[i, :])
                    policy.append((action, self.dt))

            else:
                path.append(self.key_to_state(cur_key))
                policy.append((action, t))

            cur_key = next_key
        return path, policy

    def graphics_to_cartesian_(self, state):
        offset = np.zeros_like(state)
        offset[self.get_y(index=True)] = self.graph.height
        return offset - state

    def nonholonomic_obstacle_free_(self, cur, target):
        """Since euclidean distance is not a good heuristic for non-holonomic vehicles, we use dubin's path to estimate path length. While these 
        values can be pre-computed in a heuristic lookup table (HLUT), 
        there are many positions to precompute so the time saved by this
        may not outweigh the high overhead of precomputing all state heuristics.
        Also, dubin's path is a fast computation and this library is a Cython wrapper and thus still performs quickly.

        Args:
            cur ([type]): [description]
            target ([type]): [description]

        Returns:
            [type]: [description]
        """
        cur_pose = cur[:3]  # x0, y0, theta0
        target_pose = target[:3]  # x1, y1, theta1
        # NOTE: we perform search from q0=target_pos to q1=cur_pos
        # since we are doing backward search, this is actually what's
        # going on, and this avoids issues of transforming frames
        path = dubins.shortest_path(target_pose, cur_pose, self.min_turn_rad)
        return path.path_length()

    def holonomic_obstacle_(self, cur, target):
        x0, y0, _, _, _ = self.state_to_key(cur)
        x1, y1, _, _, _ = self.state_to_key(target)
        dx = self.dstate[0]
        path, dist = self.obstacle_heuristic.find_path(
            start=(x0, y0), goal=(x1, y1))
        return dist * dx

    def heuristic(self, cur, target):
        """Since backward search, heuristic  defined wrt start.

        Args:
            state ([type]): [description]
        """
        # TODO: Euclidean distance grossly underestimates true
        # cost for dynamically-constrained vehicles, should find
        # a better heuristic
        # cur_pos = np.array(
        #     [self.get_x(state=cur),         # x
        #      self.get_y(state=cur),         # y
        #      self.graph.get_map_val(cur)])  # z

        # start_pos = np.array(
        #     [self.get_x(state=target),         # x
        #      self.get_y(state=target),         # y
        #      self.graph.get_map_val(target)])  # z

        constraints_dist_cost = self.nonholonomic_obstacle_free_(cur, target)
        obstacle_dist_cost = self.holonomic_obstacle_(cur, target)
        return max(constraints_dist_cost, obstacle_dist_cost)

    def reached_target(self, state, target):
        """Basically like found-goal, but applied to backwards search. Still using goal threshold

        Args:
            state ([type]): [description]
        """
        # reaching goal requires similar position and heading
        # TODO: also include velocity for a dynamics-considering version?
        heading_dist = abs(
            self.get_theta(state=state) - self.get_theta(state=target)) % TWO_PI
        is_similar_heading = heading_dist < self.heading_thresh
        print(self.heuristic(state, target), heading_dist * 180 / math.pi)
        is_spatially_near = self.heuristic(state, target) < self.goal_thresh

        return is_spatially_near  # and is_similar_heading

    def state_equal(self, n1, n2, ignore_time=False):
        # let state_to_key handle issues of angle wraparound
        n1_key = self.state_to_key(n1)
        n2_key = self.state_to_key(n2)
        if ignore_time:
            return n1_key[:-1] == n2_key[:-1]
        else:
            return n1_key == n2_key

    def get_value(self, val_map, *, state=None, state_key=None):
        assert(state is not None or state_key is not None)
        if state_key is None:
            state_key = self.state_to_key(state)

        if state_key in val_map:
            return val_map[state_key]
        else:
            # default cost infinity if never seen
            return self.INF

    def set_value(self, val_map, val, *, state=None, state_key=None):
        assert (state is not None or state_key is not None)
        if state_key is not None:
            val_map[state_key] = val
        else:
            val_map[self.state_to_key(state)] = val

    @ staticmethod
    def compute_f(g, h, eps):
        # just to be explicit
        return g + eps * h

    @ staticmethod
    def get_x(state=None, index=False):
        if index:
            return 0
        return state[0]

    @ staticmethod
    def get_y(state=None, index=False):
        if index:
            return 1
        return state[1]

    @ staticmethod
    def get_theta(state=None, index=False):
        if index:
            return 2
        return state[2]

    @ staticmethod
    def get_vel(state=None, index=False):
        if index:
            return 3
        return state[3]

    @ staticmethod
    def get_time(state=None, index=False):
        if index:
            return 4
        return state[4]

    @ staticmethod
    def state_to_str(state):
        return "(%.2f,%.2f,%.2f,%.2f, %.2f)" % (
            state[0], state[1], state[2], state[3], state[4])
