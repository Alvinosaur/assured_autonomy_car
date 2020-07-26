import numpy as np
import math
import matplotlib.pyplot as plt

from lattice_graph import Graph
from lattice_dstar_lite import LatticeDstarLite
from car_dynamics import Action

RAD_TO_DEG = 180 / math.pi
INCH_TO_M = 2.54 / 100.0


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
    # empty window should not cause error
    xbounds = [0, 0]
    ybounds = [0, 0]
    graph.update_map(xbounds, ybounds, obs_window=[])

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
        state=state[:3], action=action, dt=planner.dt, T=planner.T)
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


if __name__ == "__main__":
    run_all_tests()
