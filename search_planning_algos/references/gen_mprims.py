import numpy as np
import matplotlib.pyplot as plt
import math

INF = 1e10


def xdot(v, theta):
    return v * math.cos(theta)


def ydot(v, theta):
    return v * math.sin(theta)


def thetadot(v, curvature):
    if np.isclose(curvature, 0):
        return 0
    return v * curvature


def example_turn_rad():
    k = 1 / 3.0  # curvature
    v = 1.0  # velocity
    dt = 0.1
    T = 3.0  # horizon(sec)
    N = int(T / dt)
    traj = np.zeros(shape=(N, 3))  # x, y, theta
    x, y, theta = [0] * 3

    for i in range(1, N):
        theta += thetadot(v, k) * dt
        x += xdot(v, theta) * dt
        y += ydot(v, theta) * dt
        traj[i, :] = [x, y, theta]

    plt.plot(traj[:, 0], traj[:, 1])  # x v.s y

    # ts = np.linspace(start=0, stop=T - dt, num=N)
    # fig, axs = plt.subplots(3)
    # axs[0].plot(ts, traj[:, 0])  # x
    # axs[1].plot(ts, traj[:, 1])  # y
    # axs[2].plot(ts, traj[:, 2])  # theta
    plt.show()


def viz_prims(mprims, traj, ts):
    fig, axs = plt.subplots(2)
    for prim_i, (k, v) in enumerate(mprims):
        # Plot y v.s x
        label = "%.2f, %.2f" % (k, v)
        axs[0].plot(traj[:, 0, prim_i],
                    traj[:, 1, prim_i],
                    label=label)  # y v.s x

        # plot theta v.s time
        axs[1].plot(ts, traj[:, 2, prim_i], label=label)  # theta

    fig.suptitle(
        "Lattice Path Profiles for various (curvature, velocity) primitives")
    axs[0].legend(loc="upper right")
    axs[0].set(xlabel='X', ylabel='Y')

    axs[1].legend(loc="upper right")
    axs[1].set(xlabel='t', ylabel='Theta')
    plt.show()


def gen_mprims(T, dt, viz=True, theta_offset=0.0):
    """The same set of motion primitives can be applied to any given pose by rotating all paths with current heading. In this right image, theta_offset=+90º was applied. theta_offset can be changed for visualization purposes,
    but MAKE SURE to use theta_offset=0 when storing the final primitives.

    Args:
        T (float): time horizon of primitives (seconds)
        dt (float): time-discretization (seconds) for generated trajectories. Make larger to reduce computation complexity, but sacrifice accuracy of trajectory, which will weaken completeness guarantees.
        viz (bool, optional): whether to visualize. Defaults to True.
        theta_offset (float, optional): offset of primitives direction. Defaults to 0.
    """
    # All set of curvatures
    max_k = 1 / 3.0
    curvature_s = np.linspace(start=-max_k, stop=max_k, num=5, endpoint=True)

    # all velocity profiles (absolute value)
    vels = [1.0, 2.0]

    mprims = [[k, v] for k in curvature_s for v in vels]

    N = int(T / dt)
    traj = np.zeros(shape=(N, 3, len(mprims)))  # x, y, theta

    for i in range(1, N):
        for prim_i, (k, v) in enumerate(mprims):
            [x, y, theta] = traj[i - 1, :, prim_i]
            theta += thetadot(v, k) * dt
            x += xdot(v, theta) * dt
            y += ydot(v, theta) * dt
            traj[i, :, prim_i] = [x, y, theta]

    traj[:, 2, :] += theta_offset
    rot_mat = np.array([[math.cos(theta_offset), -math.sin(theta_offset)],
                        [math.sin(theta_offset), math.cos(theta_offset)]])
    for prim_i in range(len(mprims)):
        yx_coords = np.fliplr(traj[:, 0:2, prim_i])
        traj[:, 0:2, prim_i] = np.fliplr(yx_coords @ rot_mat)

    # visualize
    if viz:
        ts = np.linspace(start=0, stop=T - dt, num=N)
        viz_prims(mprims, traj, ts)

    return mprims, traj


if __name__ == "__main__":
    mprims_filename = "motion_primitives"
    mprims, prim_trajs = gen_mprims(dt=0.2, T=1.0)
    np.savez(mprims_filename, mprims=mprims,
             prim_trajs=prim_trajs, allow_pickle=True)
