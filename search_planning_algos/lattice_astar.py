import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import copy
import time

from car_dynamics import Action, Car
from lattice_graph import Graph
from lattice_dstar_lite import LatticeDstarLite
"""
Astar version of lattice_dstar_lite planner. Used to compare performance. Differs from basic_lattice_planner with extra time and velocity state and multiple velocities in actions
"""

TWO_PI = 2 * math.pi
DEBUG = True
RAD_TO_DEG = 180 / math.pi


class Node(object):
    def __init__(self, priority, state):
        self.priority = priority
        self.state = state

    def __lt__(self, other):
        return self.priority < other.priority


class LatticeAstar(LatticeDstarLite):
    # state = [x, y, theta]
    def __init__(self, graph: Graph, min_state, dstate, velocities, steer_angles, thetas, T, goal_thresh=None, eps=1.0, viz=False):
        super().__init__(graph=graph, min_state=min_state,
                         dstate=dstate, velocities=velocities,
                         steer_angles=steer_angles, thetas=thetas, T=T, goal_thresh=goal_thresh, eps=eps, viz=viz)
        self.path_actions = []
        self.path_timesteps = []

    def reset(self, start, goal):
        self.start = start
        self.goal = goal
        start_key = self.state_to_key(start)
        self.G = {start_key: 0}  # start has travel cost of 0
        self.open_set = [self.create_node(state=start)]
        self.successor = {start_key: None}  # start has no successors
        self.path = []
        self.expand_order = []
        self.path_actions = []
        self.path_timesteps = []
        self.path_i = 0

    def create_node(self, state, priority=0):
        return Node(priority=priority, state=state)

    def search(self, start, goal, obs_window, window_bounds):
        assert (self.is_valid_key(self.state_to_key(start)))
        assert(self.is_valid_key(self.state_to_key(goal)))
        # time irrelevant for goal so ignore
        need_update = self.goal is None or not self.state_equal(
            self.goal, goal, ignore_time=True)

        # update map if specified
        need_update |= self.graph.update_map(
            obs_window=obs_window, xbounds=window_bounds[0], ybounds=window_bounds[1])

        if self.viz:
            self.ax.clear()
            self.ax.imshow(self.graph.map)

        if need_update or len(self.path) == 0:
            self.reset(start, goal)
            success = self.compute_path()
            if success:
                return self.path  # , self.path_actions, self.path_timesteps
            else:
                print("Failed to find a solution!")
                return None

        else:
            self.path_i += 1
            return self.path[self.path_i:]
            # , self.path_actions[self.path_i:], self.path_timesteps[self.path_i:]

    def compute_path(self):
        reached_goal = False

        while len(self.open_set) > 0:
            node = heapq.heappop(self.open_set)
            cur_state = node.state
            # print("Expanded: %.2f, %.2f" % (cur_state[0], cur_state[1]))
            # print()
            current_key = self.state_to_key(cur_state)
            self.expand_order.append(current_key)
            if self.reached_target(cur_state, self.goal):
                self.goal = cur_state
                reached_goal = True
                break

            # get neighbor states update their values if necessary
            all_trajs, dist_costs, actions = self.graph.neighbors(
                cur_state, predecessor=True)

            # add neighbors to open set
            for ti, traj in enumerate(all_trajs):
                # iterate through trajectory and add states to open set
                # print()
                # print("Action: %s" % str(actions[ti]))
                for si in range(traj.shape[0]):
                    next_state = traj[si, :]
                    self.update_state(cur_state=cur_state,
                                      next_state=next_state,
                                      trans_cost=dist_costs[ti][si],
                                      action=actions[ti],
                                      t=si)

                    if self.viz:
                        dx, dy, _, _, _ = self.dstate
                        x_data = traj[:, 0] / dx
                        y_data = traj[:, 1] / dy
                        self.ax.scatter(x_data, y_data, s=2)
                        plt.draw()

            if self.viz:
                plt.pause(1)

        if reached_goal:
            states, actions, action_durs = self.reconstruct_path()
            self.path = states
            self.path_actions = actions
            self.path_timesteps = action_durs
            return True
        else:
            return False

    def update_state(self, cur_state, next_state, trans_cost, action, t):
        start_time = time.time()
        current_key = self.state_to_key(cur_state)
        next_key = self.state_to_key(next_state)
        new_G = self.G[current_key] + trans_cost
        if next_key not in self.G or new_G < self.G[next_key]:
            self.G[next_key] = new_G

            # f = g + eps*h
            priority = new_G + (
                self.eps * self.heuristic(next_state, self.goal))

            heapq.heappush(self.open_set, self.create_node(
                priority=priority, state=next_state))
            # si + 1 since 0th sample in trajectory is not current state
            self.successor[next_key] = (cur_state, action, t)

        end_time = time.time()
        self.update_state_time += (end_time - start_time)

    def reconstruct_path(self):
        states, actions, action_durs = [], [], []
        start_key = self.state_to_key(self.start)
        cur_key = self.state_to_key(self.goal)
        while cur_key != start_key:
            prev, action, sample_i = self.successor[cur_key]
            states.append(prev)
            actions.append(action)
            action_durs.append(sample_i * self.dt)
            cur_key = self.state_to_key(prev)

        states.reverse()
        actions.reverse()
        action_durs.reverse()
        return states, actions, action_durs


def viz_map_overlay_plan(actions, trajectories, dstate):
    dx, dy, _ = dstate
    for prim_i, action in enumerate(actions):
        # Plot y v.s x
        traj = trajectories[prim_i]
        plt.plot(traj[:, 0] / dx,
                 traj[:, 1] / dy)  # y v.s x
    plt.pause(1)


def get_obs_window_bounds(graph: Graph, state, width, height):
    xi, yi, _, _, _ = graph.discretize(state)
    half_width = int(width / 2)
    half_height = int(height / 2)
    xbounds = (max(0, xi - half_width),
               min(graph.width, xi + half_width))
    ybounds = (max(0, yi - half_height),
               min(graph.height, yi + half_height))
    return xbounds, ybounds


def simulate_plan_execution(start, goal, planner, true_map, viz=True):
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
        "Naive astar on lattice graph")
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

        path, actions, timesteps = planner.search(
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
    eps = 4
    dist_cost = 1
    time_cost = 1
    roughness_cost = 1
    cost_weights = (dist_cost, time_cost, roughness_cost)

    # define action space
    velocities = np.linspace(start=1, stop=2, num=2)
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
    planner = LatticeAstar(graph=graph, min_state=min_state, dstate=dstate,
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
