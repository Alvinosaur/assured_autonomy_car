import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import copy

"""
Basic lattice planner with constant velocity and instantaneous change in steering angle. Uses simple bicycle model. Assume no slip.
"""

TWO_PI = 2 * math.pi
DEBUG = True


class Car(object):
    def __init__(self, L=1.0, max_v=3, max_steer=math.pi/4):
        # state = [x, y, theta]
        self.M = 3  # size of state space

        self.L = 1.0  # length of car
        self.max_v = max_v
        self.max_steer = max_steer

    def rollout(self, state, v, steer, dt, T):
        assert (abs(v) <= self.max_v)
        assert(abs(steer) <= self.max_steer)
        N = int(T / float(dt))  # number of steps
        traj = np.zeros(shape=(N, self.M))
        state = np.array(state)
        for i in range(N):
            theta = state[2]

            # state derivatives
            thetadot = v * math.tan(steer) / self.L
            xdot = v * math.cos(theta)
            ydot = v * math.sin(theta)
            state_dot = np.array([xdot, ydot, thetadot])

            # update state and store
            state += state_dot * dt
            traj[i, :] = state

        return traj


class Action(object):
    def __init__(self, steer, v):
        self.steer = steer
        self.v = v

    def __repr__(self):
        return "steer, v: (%.2f, %.2f)" % (self.steer, self.v)


class BasicLattice(object):
    # state = [x, y, theta]
    def __init__(self, map, dstate, min_state, T, dt, base_vel, max_steer, goal_thresh=None, eps=1.0):
        self.map = np.array(map)
        self.Y, self.X = np.shape(self.map)
        self.Theta = round(TWO_PI / float(dstate[2]))
        self.dstate = np.array(dstate)
        self.min_state = np.array(min_state)
        if goal_thresh is not None:
            self.thresh = np.array(goal_thresh)
        else:
            self.thresh = np.array(dstate)
        self.eps = eps

        # motion primitive computation
        self.base_vel = base_vel
        self.dt = dt
        self.T = T
        self.N = round(T / dt)
        self.max_steer = max_steer

        # collision checking using car geometry
        # collision cost should be higher than any path distance travelled
        self.collision_cost = 1e5

        # precompute trajectories of motion primitives
        self.actions = []
        self.theta_to_trajs = [None] * self.Theta
        # check for collisions
        self.theta_to_swaths = [None] * self.Theta
        self.precompute_all_primitives()

    def precompute_all_primitives(self):
        """
        Assumes number of motion primitive angles = 5, and uses
        constant velocity assumption.
        Base primitives have 0 translation offset and are computed 
        from all discretized angles.

        For every discretized theta direction a car can face,
        all motion primitives are precomputed.

        A motion prim trajectory is (N x 3 x nprims) where N is length
        of one trajectory (defined by T secs and dt sec/sample), 3 is
        size of state space(x,y,theta), and nprims is number of 
        different motion primitives.
        """
        x0, y0 = 0, 0
        num_angles = 5
        steer_angles = np.linspace(
            start=-self.max_steer, stop=self.max_steer, num=num_angles)
        self.actions = [Action(v=self.base_vel, steer=steer)
                        for steer in steer_angles]
        nprims = len(self.actions)
        self.traj_sample_distances = [[0] * self.N for _ in range(nprims)]
        car = Car(max_steer=self.max_steer, max_v=self.base_vel)

        dtheta = self.dstate[2]
        for thetai in range(self.Theta):
            theta = thetai * dtheta
            state = np.array([x0, y0, theta])
            trajs = np.zeros(shape=(self.N, 3, nprims))
            # TODO: finish swath computation for collision-checking
            # swath =
            for ai, action in enumerate(self.actions):
                trajs[:, :, ai] = car.rollout(
                    state=state, v=action.v, steer=action.steer, dt=self.dt, T=self.T)

                # also precompute distance travelled cost of these primitives
                if thetai == 0:
                    dist_so_far = 0
                    for si in range(1, self.N):
                        dist_so_far += np.linalg.norm(
                            trajs[si-1, :2, ai] - trajs[si, :2, ai])
                        self.traj_sample_distances[ai][si] = dist_so_far

            self.theta_to_trajs[thetai] = trajs

    def get_neighbors(self, state):
        x, y, theta = state
        xi, yi, thetai = self.state_to_key(state)
        #  rotation already applied in precomputation, just offset with translation
        base_trajectories = self.theta_to_trajs[thetai]
        costs = []
        all_trajs = []
        for ai, action in enumerate(self.actions):
            traj = base_trajectories[:, :, ai] + np.array([x, y, 0])
            per_sample_cost = self.calc_cost(traj, action, ai)
            costs.append(per_sample_cost)
            all_trajs.append(traj)

        return all_trajs, costs, self.actions

    def state_to_key(self, state):
        # account for angle wrap-around
        # NOTE: order of conversion matters:
        # If modulo then discretize, theta can round to 2pi and
        # discretize to an invalid value thetai beyond allowed range
        # instead, need to discretize then modulo theta
        # any theta is valid really and just needs to be
        # brought in specific range to reduce state space size
        xi, yi, thetai = np.round(
            (state - self.min_state) / self.dstate).astype(int)
        thetai %= self.Theta
        return (xi, yi, thetai)  # immutable tuple as hash key

    def key_to_state(self, key):
        return (np.array(key) * self.dstate) + self.min_state

    def is_valid_state(self, state):
        return self.is_valid_key_(self.state_to_key(state))

    def check_collision(self, state):
        assert(self.is_valid_state(state))
        xi, yi, thetai = self.state_to_key(state)
        return self.map[yi, xi]

    def calc_cost(self, traj, action, ai):
        per_sample_cost = copy.copy(self.traj_sample_distances[ai])
        # along traj, if one state becomes invalid, all other successors
        # also are invalid
        now_invalid = False
        for i in range(self.N):
            if (now_invalid or not self.is_valid_state(traj[i, :]) or
                    self.check_collision(traj[i, :])):
                per_sample_cost[i] = None
                now_invalid = True
        return per_sample_cost

    def euc_dist(self, s1, s2):
        sq_error = (np.array(s1) - np.array(s2)) ** 2
        return (sq_error[0] + sq_error[1])**0.5

    def heuristic(self, state, goal):
        """Currently only consider x,y distance from goal, not heading
        since combining euclidean distance with non-distance unit
        requires tuning weights.
        """
        return self.euc_dist(state, goal)

    def reached_goal(self, state):
        error = state - self.goal
        # whether each component of state error within own threshold
        return np.all(np.abs(error[:2]) < self.thresh[:2])

    def search(self, start, goal):
        assert (self.is_valid_state(start))
        assert (self.is_valid_state(goal))
        self.start = np.array(start)
        self.goal = np.array(goal)
        frontier = []
        heapq.heappush(frontier, (0, self.start))
        start_key = self.state_to_key(start)
        came_from = {}
        came_from[start_key] = None
        cost_so_far = {}
        cost_so_far[start_key] = 0
        cost_with_priority = {}
        cost_with_priority[start_key] = 0  # heuristic(start, goal)
        expansion_order = {}
        reached_goal = False
        if DEBUG:
            plt.imshow(self.map)
            plt.ion()
            plt.show()
        i = 0
        while len(frontier) > 0:
            (_, current) = heapq.heappop(frontier)
            current_key = self.state_to_key(current)
            print("%.2f, %.2f, %d\n" %
                  (current[0], current[1], current[2] * 180 / math.pi))
            expansion_order[current_key] = i
            i += 1
            if self.reached_goal(current):
                self.goal = current
                reached_goal = True
                break

            # cost is list(list(cost per sample in each traj))
            all_trajs, costs, actions = self.get_neighbors(
                current)
            if DEBUG:
                ts = np.linspace(
                    start=self.dt, stop=self.dt + self.T, num=self.N)
                viz_map_overlay_plan(
                    actions=actions, trajectories=all_trajs, dstate=self.dstate)
            for ti, traj in enumerate(all_trajs):
                # iterate through trajectory and add states to open set
                for si in range(self.N):
                    # if illegal state in trajectory, disconsider
                    if costs[ti][si] is None:
                        continue

                    next = traj[si, :]
                    next_key = self.state_to_key(next)
                    new_cost = cost_so_far[current_key] + costs[ti][si]
                    if next_key not in cost_so_far or new_cost < cost_so_far[next_key]:
                        cost_so_far[next_key] = new_cost
                        # including heuristic makes exploration towards goal a priority
                        priority = new_cost + (
                            self.eps * self.heuristic(next, goal))
                        cost_with_priority[next_key] = priority
                        heapq.heappush(frontier, (priority, next))
                        # si + 1 since 0th sample in trajectory is not current state
                        came_from[next_key] = (current, actions[ti], si+1)

        if reached_goal:
            states, actions, action_durs = self.unpack(came_from)
            return states, actions, action_durs, cost_so_far, cost_with_priority, expansion_order
        else:
            return None

    def unpack(self, came_from):
        states, actions, action_durs = [], [], []
        start_key = self.state_to_key(self.start)
        cur_key = self.state_to_key(self.goal)
        while cur_key != start_key:
            prev, action, sample_i = came_from[cur_key]
            states.append(self.key_to_state(prev))
            actions.append(action)
            action_durs.append(sample_i * self.dt)
            cur_key = self.state_to_key(prev)

        states.reverse()
        actions.reverse()
        action_durs.reverse()
        return states, actions, action_durs

    def is_valid_key_(self, key):
        xi, yi, thetai = key
        return ((0 <= xi < self.X) and
                (0 <= yi < self.Y) and
                (0 <= thetai < self.Theta))

    def key_to_id_(self, key):
        assert(self.is_valid_key_(key))
        xi, yi, thetai = key
        return (xi * self.Y * self.Theta) + (yi * self.Theta) + thetai


def viz_map_overlay_plan(actions, trajectories, dstate):
    dx, dy, _ = dstate
    for prim_i, action in enumerate(actions):
        # Plot y v.s x
        traj = trajectories[prim_i]
        plt.plot(traj[:, 0] / dx,
                 traj[:, 1] / dy)  # y v.s x
    plt.pause(1)


def viz_prims(actions, trajectories, ts, title="", fig=None, axs=None):
    new_fig = False
    if fig is None or axs is None:
        new_fig = True
        fig, axs = plt.subplots(2)
    for prim_i, action in enumerate(actions):
        # Plot y v.s x
        traj = trajectories[prim_i]
        print(traj)
        label = "%.2f, %.2f" % (action.steer, action.v)
        axs[0].plot(traj[:, 0],
                    traj[:, 1],
                    label=label)  # y v.s x

        # plot theta v.s time
        axs[1].plot(ts, traj[:, 2], label=label)  # theta

    fig.suptitle(title)
    axs[0].legend(loc="upper right")
    axs[0].set(xlabel='X', ylabel='Y')

    axs[1].legend(loc="upper right")
    axs[1].set(xlabel='t', ylabel='Theta')
    if new_fig:
        plt.show()
    else:
        plt.pause(1)


def visualize_rollouts():
    map = np.zeros((50, 50))

    # make sure velocity on the same scale as map
    Y, X = map.shape
    dx, dy, dtheta = 0.1, 0.1, math.pi / 4
    dstate = [dx, dy, dtheta]
    Y *= dy
    X *= dx

    min_state = [0, 0, 0]
    T = 1.0
    dt = 0.1
    N = T / dt

    # one trajectory should cover 10 different states
    base_vel = 10 * (dx + dy) / 2.0

    max_steer = math.pi / 4
    planner = BasicLattice(map=map, dstate=dstate, min_state=min_state,  T=T,
                           dt=dt, base_vel=base_vel, max_steer=max_steer)
    thetas = np.linspace(start=0, stop=7*math.pi/4, num=7)
    x, y, = X/2, Y/2
    ts = np.linspace(start=0, stop=planner.T, num=planner.N)
    for theta in thetas:
        state = [x, y, theta]
        neighbors, trajectories, costs, actions = planner.get_neighbors(state)
        viz_prims(actions=actions, trajectories=trajectories, ts=ts,
                  title="Prims for theta = %d" % int(theta * 180 / math.pi))


def main():
    # load in map
    map_file = "search_planning_algos/maps/map3.npy"
    map = np.load(map_file)
    Y, X = map.shape
    dx, dy, dtheta = 0.1, 0.1, math.pi / 4
    dstate = np.array([dx, dy, dtheta])
    min_state = np.array([0, 0, 0])

    # convert to "continuous" space
    Y *= dy
    X *= dx

    # define motion primitives meta info
    T = 1.0
    dt = 0.1
    N = T / dt
    # one trajectory should cover 10 different states
    base_vel = 10 * (dx + dy) / 2.0
    max_steer = math.pi / 4

    # define start (chosen by viewing map with matplotlib)
    # start and goal defined in discrete space
    # which is why dx = dy = 1
    x0, y0, theta0 = 11, 11, -math.pi/2
    start = np.array([x0, y0, theta0]) * np.array([dx, dy, 1])

    # define goal
    xg, yg, thetag = 136, 136, 0
    goal = np.array([xg, yg, thetag]) * np.array([dx, dy, 1])
    goal_thresh = 2 * dstate

    # build planner and run
    eps = 2.0
    planner = BasicLattice(map=map, dstate=dstate, min_state=min_state,  T=T,
                           dt=dt, base_vel=base_vel, max_steer=max_steer, goal_thresh=goal_thresh, eps=eps)
    results = planner.search(start=start, goal=goal)
    if results is None:
        print("Failed to find solution!")
        exit(-1)
    (states, actions, action_durs,
        cost_so_far, cost_with_priority, expansion_order) = results

    # execute plan and visualize results
    fig, axs = plt.subplots(2)
    state = start
    car = Car(max_steer=max_steer, max_v=base_vel)
    T_offset = 0
    for si in range(len(states)):
        predicted_state = states[si]
        action = actions[si]
        duration = action_durs[si]

        # update ts for theta graph
        N = int(duration/dt)
        ts = np.linspace(start=dt, stop=duration + dt, num=N) + T_offset

        # run rollout generator with new planned action
        traj = car.rollout(
            state=state, v=action.v, steer=action.steer, dt=dt, T=duration)

        # update state
        state = traj[-1, :]

        # Plot y v.s x
        axs[0].plot(traj[:, 0] / dx,
                    traj[:, 1] / dy, 'g')
        predx, predy, _ = predicted_state
        axs[0].plot(predx / dx, predy / dy, 'ro')

        # plot theta v.s time
        axs[1].plot(ts, traj[:, 2])

        # update time series
        T_offset += duration

    # visualize map
    axs[0].imshow(map)

    fig.suptitle(
        "Final trajectory (Y,X) and plot of heading(theta) v.s time")
    axs[0].legend(loc="upper right")
    axs[0].set(xlabel='X', ylabel='Y')

    axs[1].legend(loc="upper right")
    axs[1].set(xlabel='t', ylabel='Theta')
    plt.show()


if __name__ == "__main__":
    # visualize_rollouts()
    main()
