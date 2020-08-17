import numpy as np
import math


class Graph():
    def __init__(self, map, min_state, dstate, thetas, velocities, wheel_radius, cost_weights):
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
        self.set_new_map(map)
        self.dstate = dstate
        self.min_state = min_state
        self.thetas = thetas
        self.velocities = velocities
        self.wheel_radius = wheel_radius

        # very planner-specific
        self.dist_weight, self.time_weight, self.roughness_weight = cost_weights

    def set_new_map(self, map):
        assert(isinstance(map, np.ndarray))
        self.map = map.astype(float)
        self.height, self.width = map.shape  # y, x

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

    def neighbors(self, state, predecessor=False, backwards=False):
        """Lookup associated base trajectory for this  theta heading. Make a copy  and apply translation of current state position. Should return a trajectory of valid positions. For neighbor trajectories, uses base precomputed trajectories offset with offset of current position and time.

        predecessor: if true, looking for states lead to current state rather than states that originate from current state. 

        backwards: if true, finding successor states but subtracting timesteps rather than adding. This is a result of applying backwards search rather than forward search, where we start search from goal. Goal will have timestep of 0 and expanding successors towards goal will have decreasing timesteps.

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
        if backwards:
            # negative offsets rather than positive
            base_trajs[:, self.get_time(index=True), :] *= -1
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

        assert (0 <= minxi and minxi <= maxxi and maxxi < self.width)
        assert (0 <= minyi and minyi <= maxyi and maxyi < self.height)

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

        # check if current state to first state in trajectory is infeasible
        first_state = np.copy(traj[0, :])
        if (not self.is_valid_state(orig) or
                self.is_collision(orig, first_state)):
            return per_sample_cost[0:0], traj[0:0, :]

        dist = 0

        # used to measure transition time taken
        cur_time = self.get_time(state=orig)

        # along traj, if one state becomes invalid, all other successors
        # also are invalid
        for i in range(self.N - 1):
            prev_state = np.copy(traj[i, :])
            current = np.copy(traj[i + 1, :])
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
            cost += self.time_weight * \
                abs(self.get_time(state=current) - cur_time)

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

    def is_collision(self, prev, next, num_steps=5):
        """Linearly interpolates between states and checks states that lie on path. Simply checks if change in z-value between any two states is greater than wheel radius. More advanced techniques can be used, considering heading and speed for safety checks.

        Args:
            prev ([type]): [description]
            next (function): [description]

        Returns:
            [type]: [description]
        """
        print(self.get_map_val(prev), self.get_map_val(next))
        (x0, y0) = self.get_x(prev), self.get_y(prev)
        (x1, y1) = self.get_x(next), self.get_y(next)
        interp = np.linspace(start=[x0, y0], stop=[x1, y1], num=num_steps)

        prev_interp = np.copy(prev)
        next_interp = np.copy(prev)
        for i in range(num_steps - 1):
            prev_x, prev_y = interp[i, :]
            next_x, next_y = interp[i + 1, :]
            prev_interp[:2] = [prev_x, prev_y]
            next_interp[:2] = [next_x, next_y]

            prev_z = self.get_map_val(prev_interp)
            cur_z = self.get_map_val(next_interp)

            if abs(cur_z - prev_z) > self.wheel_radius:
                return True

        # TODO Make car wheel a parameter passed in
        return False

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
