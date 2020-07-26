import numpy as np
import math
import matplotlib.pyplot as plt
import time
import typing as t

from lattice_graph import Graph
from lattice_dstar_lite import LatticeDstarLite
from lattice_astar import LatticeAstar

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


def create_identical_planners():
    dy, dx = 0.1, 0.1
    miny, minx = 0, 0

    # define car
    wheel_radius = 0  # anything in map with non-zero z-value is an obstacle

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

    # create planner
    # temporary, will be replaced during different benchmarks
    mock_map = np.zeros((1, 1))
    Dstar_graph = Graph(map=mock_map, min_state=min_state, dstate=dstate,
                        thetas=thetas, velocities=velocities, wheel_radius=wheel_radius, cost_weights=cost_weights)
    Dstar_planner = LatticeDstarLite(graph=Dstar_graph, min_state=min_state,
                                     dstate=dstate,
                                     velocities=velocities, steer_angles=steer_angles, thetas=thetas, T=T, eps=eps, viz=True)

    Astar_graph = Graph(map=mock_map, min_state=min_state, dstate=dstate,
                        thetas=thetas, velocities=velocities, wheel_radius=wheel_radius, cost_weights=cost_weights)
    Astar_planner = LatticeAstar(graph=Astar_graph, min_state=min_state,
                                 dstate=dstate,
                                 velocities=velocities, steer_angles=steer_angles, thetas=thetas, T=T, eps=eps, viz=True)

    return Dstar_planner, Astar_planner


def benchmark_plan_from_scratch(planner: t.Union[LatticeDstarLite, LatticeAstar],
                                start, goal, iters=50):
    assert(iters > 0)

    # same observation window identical to known map
    obs_width = 2
    obs_height = 2
    xbounds, ybounds = get_obs_window_bounds(
        graph=planner.graph, state=start, width=obs_width, height=obs_height)
    window_bounds = (xbounds, ybounds)
    obs_window = planner.graph.map[ybounds[0]:ybounds[1],
                                   xbounds[0]: xbounds[1]]

    total_time = 0
    total_states_expanded = 0
    for i in range(iters):
        start_time = time.time()
        res = planner.search(start=start, goal=goal,
                             obs_window=obs_window, window_bounds=window_bounds)
        end_time = time.time()
        print("Found solution!")
        dstar_runtime = end_time - start_time
        total_time += dstar_runtime
        total_states_expanded += len(planner.expand_order)

        assert (res is not None)
        # reset so doesn't just reuse original plan
        planner.set_new_goal(goal)

    return total_time / float(iters), total_states_expanded / float(iters)


def binary_map_benchmarks():
    # get planners
    Dstar_planner, Astar_planner = create_identical_planners()
    dx, dy, _, _, _ = Dstar_planner.dstate
    velocities = Dstar_planner.velocities

    # Map 1 tests
    map_file = "search_planning_algos/maps/map3.npy"
    map1 = np.load(map_file)
    # NOTE: make sure to copy map since will be destructively modified during planning and execution loop
    Dstar_planner.graph.set_new_map(map1)
    Astar_planner.graph.set_new_map(map1)

    # Map 1 test 1
    start = [50, 70, 0, velocities[0], 0] * np.array([dx, dy, 1, 1, 1])
    goal = [140, 20, -math.pi / 2, velocities[0], 0] * \
        np.array([dx, dy, 1, 1, 1])

    dstar_runtime, dstar_num_expansions = benchmark_plan_from_scratch(
        Dstar_planner, goal, start, iters=1)

    astar_runtime, astar_num_expansions = benchmark_plan_from_scratch(
        Astar_planner, start, goal, iters=1)

    print("Dstar runtime, num expansions: %.2fs, %.2f" %
          (dstar_runtime, dstar_num_expansions))

    print("Astar runtime, num expansions: %.2fs, %.2f" %
          (astar_runtime, astar_num_expansions))


if __name__ == "__main__":
    binary_map_benchmarks()
