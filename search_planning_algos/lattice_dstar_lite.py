import heapq
import numpy as np
import sys
import copy
import math
import matplotlib.pyplot as plt

from dstar_lite import Node

DEBUG = False
RAD_TO_DEG = 180.0 / math.pi
INCH_TO_M = 2.54 / 100.0

"""
Confusions:
1. how to give transition between negative and positive velocity? Currently have vel=0 option, but there  is no trajectory associated with this. Maybe just have x,y remain the same at that point, and low-level controller will naturally slow down and stay at this point more

2. currently not replanning at each step in a lattice path, just replanning at the end of the lattice path. Problems include
    1: What if overshoot goal? That means getting the neighbor of a given state cannot just be the end of that lattice path, but need to look at all intermediate states.

    2. What defines "reaching goal"? Just position distance, or also velocity and heading? If we reached goal in position but not in velocity (ie: moving too fast), then just don't treat as goal, and need to consider adjacent neighbors with adjacent velocity profiles, then planner will hopefully expand a previous state and move to goal with slower velocity.

    3. Continuing off #2, should velocity be included in heuristic? If goal vel =0, wouldn't there always be a bias of driving with velocity near 0? Would need to ensure reaching goal in minimal time is higher priority than moving at goal velocity.

    3. without replanning at every intermediate state along a lattice path, are losing a lot of possible solutions. In extreme case, if lattice paths are too large, may be impossible to exactly land on goal.

3. When we observe new terrain around robot, how to figure out which paths lie in this region? Can't just check which start/end nodes lie in region because a trajectory may cross over this region but start and stop outside it... Need an  efficient way to query all connections crossing over a given position (which will include multiple states w/ different thetas and velocities).

Differences from original dstar-lite:
- for original, I could use the same "get-neighbors" function to get both predecessors and successors. Here, however, predecessors != sucessors since theta(heading) controls which direction car is moving. We account for this by simply flipping the current heading by 180 degrees and running rollout generation to get predecessors. NOTE: Especially for 3D terrain, moving forwards and backwards over the same path is not the same (ex: fall over cliff v.s climb wall). But temporarily, this approach works with our cost functions and heuristic. If we want predecessor of current state facing forwards, need to flip theta by 180deg so state faces backwards. The our rollouts face backwards, but their orientations are now wrong (since also facing backwards when they should face forwards), so need to flip all their orientations back.

For testing with a map generated from image and picking points using matplotlib GUI, (x,y) is valid, but our orientation is wrong. Moving up screen is -90, moving down is +90deg.

On 7/19, added time to state so plan not only finds shortest-path, but also fastest path since two states with same position but diff velocities are always expanded together.

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


class Car(object):
    """Different from basic car by also tracking velocity in state

    Args:
        object ([type]): [description]
    """

    def __init__(self, L=1.0, max_v=3, max_steer=math.pi / 4):
        # state = [x, y, theta, v, t]
        self.M = 5  # size of state space

        self.L = 1.0  # length of car
        self.max_v = max_v
        self.max_steer = max_steer

    def rollout(self, state, v, steer, dt, T, t0=0):
        """Note: state at timestep 0 is NOT  original state, but next state, so original state is not part of rollout. This means
        every trajectory contains N successor states not including original state.
        Args:
            state ([type]): [description]
            v ([type]): [description]
            steer ([type]): [description]
            dt ([type]): [description]
            T ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert (abs(v) <= self.max_v)
        assert(abs(steer) <= self.max_steer)
        N = int(T / float(dt))  # number of steps
        traj = np.zeros(shape=(N, self.M))
        traj[:, 3] = v  # constant velocity throughout
        state = np.array(state[:3])
        t = t0
        for i in range(N):
            theta = state[2]

            # state derivatives
            thetadot = v * math.tan(steer) / self.L
            xdot = v * math.cos(theta)
            ydot = v * math.sin(theta)
            state_dot = np.array([xdot, ydot, thetadot])

            # update state and store
            state = state + state_dot * dt
            t += dt
            traj[i, :3] = state
            traj[i, -1] = t

        return traj


class LatticeNode(Node):
    """Same implementation as normal Node except with extended state:
    state = [x,y,theta,vel]

    Args:
        Node ([type]): [description]
    """

    def __init__(self, k1, k2, state):
        super().__init__(k1, k2, state)
        assert(len(state) == 5)  # 4 states for [x,y,theta,vel,t]

    def __repr__(self):
        return "%.2f,%.2f,%.2f,%.2f,%.2f, k1, k2: %.2f, %.2f" % (
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], k1, k2)


class Action(object):
    def __init__(self, steer, v):
        self.steer = steer
        self.v = v

    def __eq__(self, other):
        return (np.isclose(self.steer, other.steer, rtol=1e-5) and
                np.isclose(self.v, other.v))

    def __repr__(self):
        return "steer, v: (%.2f, %.2f)" % (self.steer, self.v)


class Graph(object):
    def __init__(self, map, min_state, dstate, thetas, velocities, wheel_radius, cost_weights, minx=0, miny=0):
        """Note: state at timestep 0 is NOT  original state, but next state, so original state is not part of rollout.

        Args:
            map ([type]): [description]
            dx ([type]): [description]
            dy ([type]): [description]
            thetas ([type]): [description]
            base_prims ([type]): [description]
            base_prims_trajs ([type]): [description]
            viz (bool, optional): [description]. Defaults to False.
        """
        self.map = map.astype(float)
        assert(isinstance(map, np.ndarray))
        self.height, self.width = map.shape  # y, x
        self.dstate = dstate
        self.min_state = min_state
        self.thetas = thetas
        self.velocities = velocities
        self.wheel_radius = wheel_radius

        # very planner-specific
        self.dist_weight, self.time_weight, self.roughness_weight = cost_weights

    def generate_trajectories_and_costs(self, mprim_actions,
                                        base_trajs, state0):
        # N x 3 X M for N timesteps and M primitives
        # organized as [x,y,theta]
        assert (len(base_trajs.shape) == 3)  # 3 dimensional
        self.N, state_dim, self.num_prims = base_trajs.shape
        assert(len(mprim_actions) == self.num_prims)
        assert (state_dim == 5)  # x, y, theta, v, t

        # basic primitives that will be rotated and translated
        # based on current state
        self.base_trajs = base_trajs

        # actions associated the above trajectories
        self.mprim_actions = mprim_actions

        # precompute rotated base trajectories for every discrete heading bin
        self.theta_to_trajs = [None] * len(self.thetas)

        # TODO: more accurate way is to compute for every x,y in trajectory
        # the overlapping cells of car(need width,length of car)
        # but for now just use the x,y states only

        # for every theta bin, apply rotation to base trajectories
        for ti, theta_offset in enumerate(self.thetas):
            new_trajs = np.copy(self.base_trajs)

            # offset theta simply added with base thetas
            new_trajs[:, self.get_theta(index=True), :] += theta_offset

            # create SE2 rotation using theta offset
            rot_mat = np.array([
                [math.cos(theta_offset), -math.sin(theta_offset)],
                [math.sin(theta_offset), math.cos(theta_offset)]])

            # apply rotation to every primitive's trajectory
            for ai in range(self.num_prims):  # action-index = ai
                # since trajectories stored as [x,y], flip to [y,x] to properly be rotated by SE2 matrix
                yx_coords = np.fliplr(new_trajs[:, 0:2, ai])

                # flip back to [x,y] after applying rotation
                new_trajs[:, 0:2, ai] = np.fliplr(yx_coords @ rot_mat)

            self.theta_to_trajs[ti] = new_trajs
            # self.all_disc_traj.append(
            #     self.discretize(new_trajs, is_traj=True))

    def neighbors(self, state, predecessor=False):
        """Lookup associated base trajectory for this  theta heading. Make a copy  and apply translation of current state position. Should return a trajectory of valid positions. For neighbor trajectories, uses base precomputed trajectories offset with offset of current position and time.

        Args:
            state ([type]): [description]

        Returns:
            future trajectories for current state
        """
        assert (self.is_valid_state(state))

        # Need to copy since we destructively modify state
        state = np.array(state)
        x, y, _, _, t = state

        # if obtaining predecessors, flip theta
        # to treat as searching backwards
        if predecessor:
            state[self.get_theta(index=True)] += math.pi
        _, _, theta_i, _, _ = self.discretize(state)
        assert (0 <= theta_i < len(self.thetas))
        base_trajs = np.copy(self.theta_to_trajs[theta_i])
        costs = []
        valid_trajs = []

        translation = np.array([x, y, 0, 0, t])
        for ai, action in enumerate(self.mprim_actions):
            # apply translation in position and time
            new_traj = base_trajs[:, :, ai] + translation
            # NOTE: destructively modify trajs
            per_sample_cost, new_traj = self.calc_cost_and_validate_traj(
                state, new_traj, action, ai)

            # if generating predecessors, need to flip back their thetas to original orientation that state was facing
            if predecessor:
                new_traj[:, self.get_theta(index=True)] -= math.pi

            costs.append(per_sample_cost)
            valid_trajs.append(new_traj)
        return valid_trajs, costs, self.mprim_actions
        # need to find which discrete states the trajectory covers
        # could be pre-computed when given the continuous-space trajectory?

    def update_map(self, xbounds, ybounds, obs_window):
        """obs_window is a 2D matrix holding updated z-values. Not necessarily
        centered about robot if near map edge and observation window is truncated. X and  y bounds are defined in C-space

        Args:
            center ([type]): [description]
            obs_window ([type]): [description]
        """
        minxi, maxxi = xbounds
        minyi, maxyi = ybounds

        assert (0 <= minxi and minxi < maxxi and maxxi < self.width)
        assert (0 <= minyi and minyi < maxyi and maxyi < self.height)

        # 1 cm tolerance
        need_update = not np.allclose(
            self.map[minyi: maxyi,
                     minxi: maxxi], obs_window, atol=1e-2)

        # update map
        self.map[minyi:maxyi,
                 minxi: maxxi] = obs_window
        return need_update

    def calc_cost_and_validate_traj(self, orig, traj, action, ai):
        """Destructively modify traj to only include valid  states. Usually costs should be determined by planner, not the map module, but easier to leave all costing here since cost is heavily dependent on map terrain and distance traveled, so avoid copying over map to planner.

        Args:
            current ([type]): [description]
            traj ([type]): [description]
            action ([type]): [description]
            ai ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert(self.is_valid_state(orig))
        # TODO: add slight penalty for non-straight paths?
        per_sample_cost = [0] * self.N

        prev_coord = np.array(
            [self.get_x(state=orig),   # x
             self.get_y(state=orig),   # y
             self.get_map_val(orig)])  # z
        dist = 0

        # along traj, if one state becomes invalid, all other successors
        # also are invalid
        now_invalid = False
        prev_state = np.copy(traj[0, :])
        for i in range(self.N):
            current = np.copy(traj[i, :])
            # if collide  with obstacle or move out-of-bounds, truncate and return
            if (not self.is_valid_state(current) or
                    self.is_collision(prev_state, current)):
                return per_sample_cost[0:i], traj[0:i, :]

            cost = 0
            # calculate distance-traveled cost
            cur_coord = np.array(
                [self.get_x(state=current),   # x
                 self.get_y(state=current),   # y
                 self.get_map_val(current)])  # z
            dist += np.linalg.norm(cur_coord - prev_coord)
            cost += self.dist_weight * dist

            # calculate time-based cost
            cost += self.time_weight * self.get_time(state=current)

            # store cost
            per_sample_cost[i] = cost

            # update prev state and coord
            prev_state = current
            prev_coord = cur_coord

        # no collisions or out-of-bounds, so return full costs and trajectory
        return per_sample_cost, traj

    def is_valid_state(self, state):
        xi, yi, thetai, vi, ti = self.discretize(state)
        return (0 <= xi < self.width and
                0 <= yi < self.height and
                0 <= thetai < len(self.thetas) and
                0 <= vi < len(self.velocities) and
                0 <= ti)

    def is_collision(self, prev, next):
        prev_z = self.get_map_val(prev)
        cur_z = self.get_map_val(next)
        # TODO Make car wheel a parameter passed in
        return abs(cur_z - prev_z) > self.wheel_radius

    def get_map_val(self, state):
        assert(self.is_valid_state(state))
        xi, yi, _, _, _ = self.discretize(state)
        return self.map[yi, xi]

    def discretize(self, state):
        state_key = np.round(
            (np.array(state) - self.min_state) / self.dstate).astype(int)

        # wrap around theta back to [0, 2pi]
        state_key[self.get_theta(index=True)] %= len(self.thetas)
        return tuple(state_key)

    def make_continuous(self, disc_state):
        return (np.array(disc_state) * self.dstate) + self.min_state

    # reduce chance of index bugs
    @staticmethod
    def get_x(state=None, index=False):
        if index:
            return 0
        return state[0]

    @staticmethod
    def get_y(state=None, index=False):
        if index:
            return 1
        return state[1]

    @staticmethod
    def get_theta(state=None, index=False):
        if index:
            return 2
        return state[2]

    @staticmethod
    def get_vel(state=None, index=False):
        if index:
            return 3
        return state[3]

    @staticmethod
    def get_time(state=None, index=False):
        if index:
            return 4
        return state[4]

    @staticmethod
    def state_to_str(state):
        return "(%.2f,%.2f,%.2f,%.2f, %.2f)" % (
            state[0], state[1], state[2], state[3], state[4])


class LatticeDstarLite(object):
    INF = 1e10
    MAX_THETA = 2 * math.pi

    def __init__(self, graph: Graph, min_state, dstate, velocities, steer_angles, thetas, T, goal_thresh=None, eps=1.0, viz=False):
        self.min_state = min_state
        self.dstate = dstate
        self.velocities = velocities
        self.thetas = thetas
        self.steer_angles = steer_angles
        self.viz = viz
        if viz:
            f = plt.figure()
            self.ax = f.add_subplot(111)
            self.ax.imshow(graph.map)
            plt.ion()
            plt.show()

        # define action space
        self.graph = graph
        self.actions = [Action(v=v, steer=steer)
                        for steer in steer_angles for v in velocities]
        self.dt = self.get_time(state=dstate)
        self.T = T
        self.N = int(self.T / self.dt)
        self.precompute_all_primitives()

        if goal_thresh is None:
            self.goal_thresh = np.linalg.norm(self.dstate[:2])
        else:
            self.goal_thresh = goal_thresh
        self.heading_thresh = self.get_theta(state=dstate)

        self.is_new = True
        self.shifted_start = False
        self.eps = eps

        # values that can change if new goal is set
        self.km = 0  # changing start position and thus changing heuristic

        # populated during call to self.search()
        self.successor = dict()
        self.path = []
        self.start = self.goal = None

        # WIDTH x HEIGHT x Thetas x Vels
        self.state_space_dim = (self.graph.width, self.graph.height,
                                len(self.thetas), len(self.velocities))
        # G and V values explained in dstar_lite.py
        self.G = dict()
        self.V = dict()

    def precompute_all_primitives(self):
        # not a fully-defined state, but used as origin
        x0, y0, theta0 = 0, 0, 0
        state0 = np.array([x0, y0, theta0])
        self.car = Car(max_steer=max(self.steer_angles),
                       max_v=max(self.velocities))

        # N x 5 X M for N timesteps and M primitives
        self.base_trajs = np.zeros(
            shape=(self.N, len(self.dstate), len(self.actions)))
        for ai, action in enumerate(self.actions):
            self.base_trajs[:, :, ai] = self.car.rollout(
                state=state0, v=action.v, steer=action.steer, dt=self.dt, T=self.T, t0=0)

        self.graph.generate_trajectories_and_costs(
            mprim_actions=self.actions, base_trajs=self.base_trajs, state0=state0)

    def set_new_goal(self, goal):
        self.goal = goal
        self.V = dict()
        self.G = dict()
        self.set_value(self.V, goal, 0)
        self.open_set = [self.create_node(goal)]
        self.successor = dict()
        self.successor[self.state_to_key(goal)] = None
        self.path = []
        self.km = 0

    def compute_path_with_reuse(self):
        expand_order = []

        # update whether start is inconsistent
        gstart = self.get_value(self.G, self.start)
        vstart = self.get_value(self.V, self.start)
        start_inconsistent = not np.isclose(gstart, vstart, atol=1e-5)
        if len(self.open_set) == 0:
            return expand_order

        start_node = self.create_node(self.start)
        min_node = self.open_set[0]

        while len(self.open_set) > 0 and (
                (min_node < start_node) or start_inconsistent):
            # print("(Fstart, Fmin): (%s, %s)" % (start_node, min_node))
            # expand next node w/ lowest f-cost, add to closed
            # print(self.open_set)
            cur_node = heapq.heappop(self.open_set)
            # print("   Expanded %d : %s: %s" % (expand_i, str(cur_node),
            #                                    str(list(self.graph.neighbors(cur_node.state)))))
            # print("Current: %s" % cur_node)
            cur_state = cur_node.state
            print("Expanded: %s" % self.state_to_str(cur_state))

            # track order of states expanded
            expand_order.append(cur_state)

            # update current state's G value with V
            g_cur = self.get_value(self.G, cur_state)
            v_cur = self.get_value(self.V, cur_state)
            if g_cur > v_cur:
                # g_cur = v_cur
                self.set_value(self.G, cur_state, v_cur)
            else:
                self.set_value(self.G, cur_state, self.INF)
                self.update_state(cur_state)  # Pred(s) U {s}

            # if reached start target state, update fstart value
            # implement goal threshold by applying threshold to start since backwards search. If any state is within threshold of start, just pick this state as new start.
            if self.found_start(cur_state):
                # time of course won't be the same
                if not self.state_equal(self.start, cur_state, ignore_time=True):
                    self.km += self.heuristic(self.start, cur_state)
                    self.start = cur_state
                    start_node = self.create_node(self.start)
                    # return entire path, not just path[1:]
                    self.shifted_start = True

            # get neighbor states update their values if necessary
            all_trajs, dist_costs, actions = self.graph.neighbors(
                cur_state, predecessor=True)
            for ti, traj in enumerate(all_trajs):
                # iterate through trajectory and add states to open set
                print()
                print("Action: %s" % str(actions[ti]))
                for si in range(traj.shape[0]):
                    next = traj[si, :]
                    self.update_state(next)

                if self.viz:
                    dx, dy, _, _, _ = self.dstate
                    x_data = traj[:, 0] / dx
                    y_data = traj[:, 1] / dy
                    self.ax.scatter(x_data, y_data, s=2)
                    plt.draw()

            if self.viz:
                plt.pause(1)

            # TODO: if reached start target state, update fstart value? Original dstar_lite doesn't do anything meaningful here

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
        return expand_order

    def remove_from_open(self, target_state):
        for i in range(len(self.open_set)):
            node = self.open_set[i]
            if self.state_equal(node.state, target_state):
                # set node to remove as last element, remove duplicate, reorder
                self.open_set[i] = self.open_set[-1]
                self.open_set.pop()
                heapq.heapify(self.open_set)
                return

    def search(self, start, goal, obs_window, window_bounds):
        self.shifted_start = False  # reset
        if self.viz:
            self.ax.clear()
            self.ax.imshow(self.graph.map)
        if self.start is not None:
            self.km += self.heuristic(self.start, start)
        self.start = start
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
        if need_update:
            # point of dstar lite is to not update every state in open-set
            # but only those that are relevant, so  only update start and let this propagate
            self.update_state(self.start)

        # reverse start and goal so search backwards
        expansions = self.compute_path_with_reuse()
        # print("Order of expansions:")
        # viz.draw_grid(self.graph, width=3, number=expansions, start=start,
        #               goal=self.goal)
        self.path = self.reconstruct_path()
        return self.path

    def update_closest_to_start(self, cur_state):
        # TODO: Currently unused
        dist = np.linalg.norm(cur_state[:2] - self.start[:2])
        if dist < self.best_dist:
            self.best_dist = dist
            self.closest_to_start = np.copy(cur_state)

    def get_min_g_val(self, cur):
        min_g = self.INF
        best_neighbor = None
        all_trajs, trans_cost, actions = self.graph.neighbors(cur)
        for ti, traj in enumerate(all_trajs):
            # iterate through trajectory and add states to open set
            for si in range(traj.shape[0]):
                next = traj[si, :]
                cost = (trans_cost[ti][si] + self.get_value(self.G, next))
                if cost < min_g or best_neighbor is None:
                    min_g = cost
                    best_neighbor = next

        return min_g, best_neighbor

    def create_node(self, cur):
        k1, k2 = self.get_k(cur)
        return LatticeNode(k1, k2, cur)

    def get_k(self, cur):
        v = self.get_value(self.V, cur)
        g = self.get_value(self.G, cur)
        k2 = min(v, g)
        # k1 = f-value
        h = self.heuristic(cur)
        # km here accounts for issue where the moving start is our "goal" with
        # backward search, so the heuristics always change, but can just add
        # constant
        k1 = self.compute_f(g=k2, h=h + self.km, eps=self.eps, )
        print("%s: g(%.2f) + eps*h(%.2f) = (%.2f):" % (
            self.state_to_str(cur), k2, k1 - k2, k1))
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

    def update_state(self, cur_state):
        # if already in openset, need to remove since has outdated f-val
        self.remove_from_open(cur_state)

        # get updated g-value of current state
        if not self.state_equal(cur_state, self.goal, ignore_time=True):
            min_g, best_neighbor = self.get_min_g_val(cur_state)
            self.set_value(self.V, cur_state, min_g)
            self.successor[self.state_to_key(
                cur_state)] = self.state_to_key(best_neighbor)

        # if inconsistent, insert into open set
        v = self.get_value(self.V, cur_state)
        g = self.get_value(self.G, cur_state)
        if not np.isclose(v, g, atol=1e-5):
            heapq.heappush(self.open_set, self.create_node(cur_state))

    def reconstruct_path(self):
        """Returns a path ordered from start to goal. If found state close to,
        but not exactly the start state, then shifted start is at path[0]. Otherwise, start is not part of path and path[0] is next state. Goal is at path[-1].

        Returns:
            [type]: [description]
        """
        cur_key = self.state_to_key(self.start)
        goal_key = self.state_to_key(self.goal)
        path = []
        if self.shifted_start:
            # add start if this start isn't exactly original start
            path.append(self.start)
        while cur_key != goal_key:
            cur_key = self.successor[cur_key]
            path.append(self.key_to_state(cur_key))
        return path

    def heuristic(self, cur):
        """Since backward search, heuristic  defined wrt start.

        Args:
            state ([type]): [description]
        """
        # TODO: Euclidean distance grossly underestimates true
        # cost for dynamically-constrained vehicles, should find
        # a better heuristic
        cur_pos = np.array(
            [self.get_x(state=cur),         # x
             self.get_y(state=cur),         # y
             self.graph.get_map_val(cur)])  # z
        start_pos = np.array(
            [self.get_x(state=self.start),         # x
             self.get_y(state=self.start),         # y
             self.graph.get_map_val(self.start)])  # z

        dist = np.linalg.norm(start_pos - cur_pos)
        return dist

    def found_start(self, state):
        """Basically like found-goal, but applied to backwards search. Still using goal threshold

        Args:
            state ([type]): [description]
        """
        # reaching goal requires similar position and heading
        # TODO: also include velocity for a dynamics-considering version?
        heading_dist = abs(
            self.get_theta(state=state) - self.get_theta(state=self.start))
        is_similar_heading = heading_dist < self.heading_thresh

        is_spatially_near = self.heuristic(state) < self.goal_thresh

        return is_spatially_near and is_similar_heading

    def state_equal(self, n1, n2, ignore_time=False):
        # let state_to_key handle issues of angle wraparound
        n1_key = self.state_to_key(n1)
        n2_key = self.state_to_key(n2)
        if ignore_time:
            return n1_key[:-1] == n2_key[:-1]
        else:
            return n1_key == n2_key

    def get_value(self, val_map, state):
        return val_map.get(self.state_to_key(state), self.INF)

    def set_value(self, val_map, state, val):
        val_map[self.state_to_key(state)] = val

    @ staticmethod
    def compute_f(g, h, eps):
        # just to be explicit
        return g + eps * h

    @staticmethod
    def get_x(state=None, index=False):
        if index:
            return 0
        return state[0]

    @staticmethod
    def get_y(state=None, index=False):
        if index:
            return 1
        return state[1]

    @staticmethod
    def get_theta(state=None, index=False):
        if index:
            return 2
        return state[2]

    @staticmethod
    def get_vel(state=None, index=False):
        if index:
            return 3
        return state[3]

    @staticmethod
    def get_time(state=None, index=False):
        if index:
            return 4
        return state[4]

    @staticmethod
    def state_to_str(state):
        return "(%.2f,%.2f,%.2f,%.2f, %.2f)" % (
            state[0], state[1], state[2], state[3], state[4])


def get_obs_window_bounds(graph: Graph, state, width, height):
    xi, yi, _, _, _ = graph.discretize(state)
    half_width = int(width / 2)
    half_height = int(height / 2)
    xbounds = (max(0, xi - half_width),
               min(graph.width, xi + half_width))
    ybounds = (max(0, yi - half_height),
               min(graph.height, yi + half_height))
    return xbounds, ybounds


def run_all_tests():
    """
    test upper and lower bounds of half bins
    test that out-of-bound assertions are called
    """
    # define car
    wheel_radius = 24 / 2.0 * INCH_TO_M

    # define position space
    dy, dx = 0.1, 0.1
    miny, minx = -2, -2
    maxy, maxx = 8, 8
    Y = int(round((maxy - miny + dy) / dy))  # [min, max] inclusive
    X = int(round((maxx - minx + dx) / dx))
    map = np.zeros((Y, X))

    # define action space
    velocities = np.linspace(start=1, stop=3, num=3)
    dv = velocities[1] - velocities[0]
    dt = 0.1
    T = 1.0
    steer_angles = np.linspace(-math.pi / 4, math.pi / 4, num=5)

    # define heading space
    start, stop, step = 0, 315, 45
    num_thetas = int((stop - start) / step) + 1
    thetas = np.linspace(start=0, stop=315, num=num_thetas)
    # NOTE: Actual thetas are in radians! here just shown in degrees for clarity
    assert (np.allclose(thetas, [0, 45, 90, 135, 180, 225, 270, 315]))
    thetas = thetas / RAD_TO_DEG  # convert to radians
    dtheta = step / RAD_TO_DEG

    # collective variables for discretizing C-sapce
    dstate = np.array([dx, dy, dtheta, dv, dt])
    min_state = np.array([minx, miny, min(thetas), min(velocities), 0])

    # arbitrary
    start = [0, 0, 0, 0, 0]  # time doesn't matter here
    goal = [5, 5, 0, 0, 0]  # time doesn't matter here
    cost_weights = [1, 1, 1]
    graph = Graph(map=map, min_state=min_state, dstate=dstate,
                  thetas=thetas, velocities=velocities, wheel_radius=wheel_radius, cost_weights=cost_weights)
    planner = LatticeDstarLite(graph=graph, min_state=min_state, dstate=dstate,
                               velocities=velocities, steer_angles=steer_angles, thetas=thetas, T=T)

    # irrelevant for time
    max_state = np.array([maxx, maxy, thetas[-1], velocities[-1], 0])
    test_state_equal(min_state=min_state, max_state=max_state, dstate=dstate,
                     thetas=thetas, velocities=velocities, planner=planner, graph=graph)

    test_graph_update_map(graph, min_state, max_state)
    test_graph_traj_costing(graph, planner, max_state)
    test_visualize_primitives(graph)


def test_state_equal(min_state, max_state, dstate, thetas,
                     velocities, planner, graph):
    # test that upper and lower bounds map to same key for different states
    test_states = [min_state, max_state]
    for state in test_states:
        s1 = np.array(state)
        s1_key = planner.state_to_key(s1)
        # subtract 1e-10 to break tie (ie: 5.5 -> 6, but 5.499 -> 5) for python2.7
        # NOTE: using hard-coded 1e-10 to break ties may cause bug if level
        # of discretization is finer than 1e-10
        upper_s1 = s1 + (dstate / 2.0) - 1e-10
        lower_s1 = s1 - (dstate / 2.0) + 1e-10
        assert (s1_key == planner.state_to_key(
            upper_s1) == graph.discretize(upper_s1))
        assert(s1_key == planner.state_to_key(
            lower_s1) == graph.discretize(lower_s1))

    # make it explicit how rounding is done, very hard-coded
    s1 = np.array([8, 8, 135 / RAD_TO_DEG, 2, 0])
    # with python 3, 0.5 rounds down to 0, so make tests work for both versions
    upper_s1 = s1 + \
        np.array([0.0499, 0.0499, 22.499 / RAD_TO_DEG, 0.499, 0.0499])
    assert (planner.state_equal(s1, upper_s1))
    past_s1 = s1 + np.array([0.051, 0.051, 22.51 / RAD_TO_DEG, 0.51, 0.051])
    s1_key = np.array(planner.state_to_key(s1))
    try:
        past_s1_key = np.array(planner.state_to_key(past_s1))
        print("Error: Values should be out-of-bounds and raise assertion!")
        assert(False)
    except:
        pass

    # show that time can be ignored for defining goal since unknown
    s1[-1] = 95
    assert (not planner.state_equal(s1, upper_s1))
    assert (planner.state_equal(s1, upper_s1, ignore_time=True))


def test_graph_update_map(graph: Graph, min_state, max_state):
    dx, dy, _, _, _ = graph.dstate
    # single-value window at very edge of map should not cause error
    xbounds = [0, 1]
    ybounds = [0, 1]
    graph.update_map(xbounds, ybounds, obs_window=np.zeros((1, 1)))

    # near min and max with window extending to edge
    # state is diagonal neighbor of map's corner, window size limited to 1
    #   |
    #   |
    #   |  x
    #   |________
    width = 5
    height = 8
    xbounds = [0, width]
    ybounds = [0, height]
    window = np.ones((height, width))
    need_update = graph.update_map(xbounds, ybounds, obs_window=window)
    assert (need_update)
    assert(np.allclose(graph.map[0:height, 0:width], window))
    # reset map to not mess up other tests
    graph.map.fill(0)


def test_graph_traj_costing(graph: Graph, planner: LatticeDstarLite, max_state):
    # create obstacle in map
    dx = planner.dstate[0]
    obst_xi = 20
    min_x, _, _, _, _ = planner.key_to_state(
        [obst_xi, 0, planner.thetas[0], planner.velocities[0], 0])
    graph.map[:, obst_xi:obst_xi + 5] = graph.wheel_radius + \
        1  # large jump along x = 5

    # facing obstacle at xi = 5
    state = [min_x - 10 * dx, 0, 0, planner.get_vel(state=max_state), 0]
    action = Action(steer=0, v=planner.get_vel(state=max_state))
    ai = planner.actions.index(action)
    traj = planner.car.rollout(
        state=state[:3], v=action.v, steer=action.steer, dt=planner.dt, T=planner.T)
    costs, valid_traj = graph.calc_cost_and_validate_traj(
        state, traj, action=action, ai=ai)
    # trajectory should have been truncated since collides with obstacle
    assert (valid_traj.shape[0] < traj.shape[0])
    assert (valid_traj[-1, 0] < 5)  # valid traj stops before obstacle


def test_visualize_primitives(graph):
    fig, axs = plt.subplots(2)
    plt.xlabel('x')
    plt.ylabel('y')
    # visualize successors and predecessors
    state_discr = [graph.width / 2, graph.height / 2, 0, 1, 0]
    state = graph.make_continuous(state_discr)
    for theta in graph.thetas:
        print("Theta: %d " % (theta * 180 / math.pi))
        state[2] = theta
        successors, _, _ = graph.neighbors(
            state, predecessor=False)

        predecessors, _, _ = graph.neighbors(
            state, predecessor=True)

        for i in range(len(successors)):
            succ_traj = successors[i]
            pred_traj = predecessors[i]
            # first plot for successors
            axs[0].plot(succ_traj[:, 0],  # x
                        succ_traj[:, 1])  # y

            # second plot for predecessors
            axs[1].plot(pred_traj[:, 0],  # x
                        pred_traj[:, 1])  # y

        plt.pause(2)
        axs[0].clear()
        axs[1].clear()


def simulate_plan_execution(start, goal, planner: LatticeDstarLite, true_map, viz=True):
    dx, dy, _, _, _ = planner.dstate
    obs_width = 5
    obs_height = 5

    # two plots, one for true map, one for known map
    f = plt.figure()
    axs0 = f.add_subplot(211)
    axs1 = f.add_subplot(212)
    axs0.imshow(true_map)
    axs1.imshow(planner.graph.map)
    f.suptitle(
        "Interleaved planning and execution with dstar on lattice graph")
    axs0.set_title("True map")
    axs0.legend(loc="upper right")
    axs0.set(xlabel='X', ylabel='Y')

    axs1.set_title("Known map")
    axs1.legend(loc="upper right")
    axs1.set(xlabel='X', ylabel='Y')

    # run interleaved execution and planning
    while not planner.state_equal(start, goal, ignore_time=True):
        # make observation around current state
        xbounds, ybounds = get_obs_window_bounds(
            graph=planner.graph, state=start, width=obs_width, height=obs_height)
        obs_window = true_map[ybounds[0]:ybounds[1],
                              xbounds[0]: xbounds[1]]

        path = planner.search(
            start=start, goal=goal, obs_window=obs_window, window_bounds=(xbounds, ybounds))

        # visualize stuff
        if viz:
            axs0.clear()
            axs1.clear()
            # Plot y v.s x
            viz_path = np.array(path)
            # show planned path on both maps
            axs0.plot(viz_path[:, 0] / dx,
                      viz_path[:, 1] / dy, c='g', markersize=2)
            axs1.plot(viz_path[:, 0] / dx,
                      viz_path[:, 1] / dy, c='g', markersize=2)

            # show updated known map
            axs1.imshow(planner.graph.map)
            # need to show true map too because we clear axes
            # to show updated path
            axs0.imshow(true_map)
            plt.pause(1)

        start = path[0]


def main():
    # load in map
    map_file = "search_planning_algos/maps/map3.npy"
    map = np.load(map_file)
    Y, X = map.shape
    dy, dx = 0.1, 0.1
    miny, minx = 0, 0

    # define car
    wheel_radius = 0  # anything non-zero is an obstacle

    # define search params
    eps = 1
    dist_cost = 1
    time_cost = 1
    roughness_cost = 1
    cost_weights = (dist_cost, time_cost, roughness_cost)

    # define action space
    velocities = np.linspace(start=1, stop=3, num=3)
    dv = velocities[1] - velocities[0]
    dt = 0.1
    T = 1.0
    steer_angles = np.linspace(-math.pi / 4, math.pi / 4, num=3)

    # define heading space
    start, stop, step = 0, 315, 45
    num_thetas = int((stop - start) / step) + 1
    thetas = np.linspace(start=0, stop=315, num=num_thetas)
    thetas = thetas / RAD_TO_DEG  # convert to radians
    dtheta = step / RAD_TO_DEG

    # collective variables for discretizing C-sapce
    dstate = np.array([dx, dy, dtheta, dv, dt])
    min_state = np.array([minx, miny, min(thetas), min(velocities), 0])

    # create planner and graph
    graph = Graph(map=map, min_state=min_state, dstate=dstate,
                  thetas=thetas, velocities=velocities, wheel_radius=wheel_radius, cost_weights=cost_weights)
    planner = LatticeDstarLite(graph=graph, min_state=min_state, dstate=dstate,
                               velocities=velocities, steer_angles=steer_angles, thetas=thetas, T=T, eps=eps, viz=True)

    # define start and  goal (x,y) need to be made continuous
    # since I selected those points on image map of discrete space
    start = [50, 70, 0, velocities[0], 0] * np.array([dx, dy, 1, 1, 1])
    # looks like goal should face up, but theta is chosen
    # in image-frame as is the y-coordinates, so -90 faces
    # upwards on our screen and +90 faces down
    goal = [140, 20, -math.pi / 2, velocities[0], 0] * \
        np.array([dx, dy, 1, 1, 1])

    # run planner
    simulate_plan_execution(start=start, goal=goal,
                            planner=planner, true_map=map)


if __name__ == "__main__":
    main()
    # run_all_tests()
