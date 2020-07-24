import numpy as np
import math


class Action():
    def __init__(self, steer, v):
        self.steer = steer
        self.v = v

    def __eq__(self, other):
        return (np.isclose(self.steer, other.steer, rtol=1e-5) and
                np.isclose(self.v, other.v))

    def __repr__(self):
        return "steer, v: (%.2f, %.2f)" % (self.steer, self.v)


class Car():
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

    def rollout(self, state, action: Action, dt, T, t0=0):
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
        v = action.v
        steer = action.steer
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
