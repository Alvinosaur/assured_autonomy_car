import numpy as np
import math
import matplotlib.pyplot as plt

from lattice_graph import Graph
from lattice_dstar_lite import LatticeDstarLite


RAD_TO_DEG = 180 / math.pi


def get_obs_window_bounds(graph: Graph, state, width, height):
    xi, yi, _, _, _ = graph.discretize(state)
    half_width = int(width / 2)
    half_height = int(height / 2)
    xbounds = (max(0, xi - half_width),
               min(graph.width, xi + half_width))
    ybounds = (max(0, yi - half_height),
               min(graph.height, yi + half_height))
    return xbounds, ybounds


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
