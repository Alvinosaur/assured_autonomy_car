import numpy as np
import math
import time
import yaml
import os
import pickle

from car_dynamics import Car, Action
from lattice_graph import Graph
from lattice_astar import LatticeAstar

RAD_TO_DEG = 180 / math.pi
PLANNER_FILE = "search_planninng_algos/planner.obj"
CAR_PARAMS_FILE = ""


def create_planner(cost_weights, thetas, steer_angles, velocities, car,
                   min_state, dstate, prior_map, eps, T):
    # create planner
    Astar_graph = Graph(map=prior_map, min_state=min_state, dstate=dstate,
                        thetas=thetas, wheel_radius=car.wheel_radius, cost_weights=cost_weights)
    astar_planner = LatticeAstar(graph=Astar_graph, car=car,
                                 min_state=min_state,
                                 dstate=dstate,
                                 velocities=velocities, steer_angles=steer_angles, thetas=thetas, T=T, eps=eps, viz=True)
    return astar_planner


def init_planner(prior_map):
    # load in vehicle params
    with open(CAR_PARAMS_FILE, 'r') as f:
        car_params = yaml.load(f)
    # define search params
    eps = 2.0
    dist_cost = 1
    time_cost = 1
    roughness_cost = 1
    cost_weights = (dist_cost, time_cost, roughness_cost)

    # Define action space
    dt = 0.1
    T = 1.0
    # velocities
    # TODO: Let handle negative velocities, but enforce basic dynamic windowing
    max_speed = car_params["max_speed"]
    num_speed = 3
    velocities = np.linspace(start=0, stop=max_speed, num=num_speed) / dt
    dv = velocities[1] - velocities[0]
    # steer angles
    max_abs_steer = car_params["max_steer"]
    num_steer = 3
    steer_angles = np.linspace(-max_abs_steer, max_abs_steer, num=num_steer)

    # create simple car model for planning
    mid_to_wheel_length = car_params["car_length"] / 2.0
    car = Car(L=mid_to_wheel_length,
              max_v=car_params["max_speed"],
              max_steer=car_params["max_steer"],
              wheel_radius=car_params["wheel_radius"])

    # define heading space
    start, stop, step = 0, 315, 45
    num_thetas = int((stop - start) / step) + 1
    thetas = np.linspace(start=0, stop=315, num=num_thetas)
    thetas = thetas / RAD_TO_DEG  # convert to radians
    dtheta = step / RAD_TO_DEG

    # collective variables for discretizing C-sapce
    dy, dx = 1.0, 1.0
    miny, minx = 0, 0
    dstate = np.array([dx, dy, dtheta, dv, dt])
    min_state = np.array([minx, miny, min(thetas), min(velocities), 0])

    # get planners
    planner = create_planner(cost_weights=cost_weights,
                             thetas=thetas,
                             steer_angles=steer_angles,
                             velocities=velocities,
                             car=car,
                             min_state=min_state,
                             dstate=dstate,
                             prior_map=prior_map,
                             eps=eps,
                             T=T)

    # store planner in file to be used for planning
    with open(PLANNER_FILE, "w") as f:
        pickle.dump(planner, f)


def sim_to_planner_state(sim_state, sim_vel, t, car_params):
    x = sim_state[car_params["xi"]]
    y = sim_state[car_params["yi"]]
    theta = sim_state[car_params["yaw_i"]]

    # steer currently unused, but should be eventually included for more
    # dynamically feasible high-level plans
    steer = sim_state[car_params["steer_i"]]
    v = sim_vel[car_params["vel_i"]]
    return np.array([x, y, theta, v, t])


def run_planner(cur_sim_state, cur_sim_vel, t, goal_sim_state, goal_sim_vel,
                true_map_window, window_bounds):
    # load car parameters
    with open(CAR_PARAMS_FILE, 'r') as f:
        car_params = yaml.load(f)

    # load planner from file
    with open(PLANNER_FILE, "r") as f:
        planner = pickle.load(f)

    # start is current state
    start = sim_to_planner_state(sim_state=cur_sim_state,
                                 sim_vel=cur_sim_vel,
                                 t=t,
                                 car_params=car_params)

    goal = sim_to_planner_state(sim_state=goal_sim_state,
                                sim_vel=goal_sim_vel,
                                t=0,  # time of goal is not used
                                car_params=car_params)

    path, policy = planner.search(
        start=start, goal=goal, obs_window=true_map_window, window_bounds=window_bounds)

    return path, policy
