import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))

import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Obstacle:
    x: float
    y: float
    rx: float
    ry: float


@dataclass
class MPCConfig:
    dt: float = 0.2
    horizon: int = 20
    wheelbase: float = 2.8
    road_half_width: float = 4.0
    v_min: float = 1.0
    v_max: float = 12.0
    a_min: float = -3.5
    a_max: float = 2.5
    delta_min: float = -0.55
    delta_max: float = 0.55
    jerk_limit: float = 1.4
    steer_rate_limit: float = 0.22
    steps: int = 44


def reference_lane(x):
    term_1 = 2.1 * 0.5 * (np.tanh((x - 12.0) / 3.0) - np.tanh((x - 24.0) / 3.0))
    term_2 = 2.4 * 0.5 * (np.tanh((x - 30.0) / 3.0) - np.tanh((x - 42.0) / 3.0))
    return term_1 - term_2


def reference_speed(x):
    slowdown_1 = 2.3 * np.exp(-((x - 17.0) / 5.5) ** 2)
    slowdown_2 = 2.8 * np.exp(-((x - 35.0) / 5.0) ** 2)
    return 9.0 - slowdown_1 - slowdown_2


def obstacle_ellipses():
    return [
        Obstacle(x=20.0, y=0.0, rx=3.0, ry=1.3),
        Obstacle(x=36.0, y=2.0, rx=2.8, ry=1.25),
    ]


class BicycleMPC:
    def __init__(self, config: MPCConfig, obstacles):
        self.cfg = config
        self.obstacles = obstacles
        self.nx = 4
        self.nu = 2
        self._build_solver()

    def _lane_reference_symbolic(self, x):
        term_1 = 2.1 * 0.5 * (ca.tanh((x - 12.0) / 3.0) - ca.tanh((x - 24.0) / 3.0))
        term_2 = 2.4 * 0.5 * (ca.tanh((x - 30.0) / 3.0) - ca.tanh((x - 42.0) / 3.0))
        return term_1 - term_2

    def _lane_reference_slope_symbolic(self, x):
        sech_sq_1a = 1.0 / ca.cosh((x - 12.0) / 3.0) ** 2
        sech_sq_1b = 1.0 / ca.cosh((x - 24.0) / 3.0) ** 2
        sech_sq_2a = 1.0 / ca.cosh((x - 30.0) / 3.0) ** 2
        sech_sq_2b = 1.0 / ca.cosh((x - 42.0) / 3.0) ** 2
        term_1 = 2.1 * 0.5 * ((1.0 / 3.0) * sech_sq_1a - (1.0 / 3.0) * sech_sq_1b)
        term_2 = 2.4 * 0.5 * ((1.0 / 3.0) * sech_sq_2a - (1.0 / 3.0) * sech_sq_2b)
        return term_1 - term_2

    def _speed_reference_symbolic(self, x):
        slowdown_1 = 2.3 * ca.exp(-((x - 17.0) / 5.5) ** 2)
        slowdown_2 = 2.8 * ca.exp(-((x - 35.0) / 5.0) ** 2)
        return 9.0 - slowdown_1 - slowdown_2

    def _dynamics(self, x, u):
        px, py, psi, v = x[0], x[1], x[2], x[3]
        a, delta = u[0], u[1]
        return ca.vertcat(
            v * ca.cos(psi),
            v * ca.sin(psi),
            v * ca.tan(delta) / self.cfg.wheelbase,
            a,
        )

    def _rk4_step(self, x, u):
        dt = self.cfg.dt
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * dt * k1, u)
        k3 = self._dynamics(x + 0.5 * dt * k2, u)
        k4 = self._dynamics(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_solver(self):
        N = self.cfg.horizon
        nu = self.nu
        n_obstacles = len(self.obstacles)

        U = ca.SX.sym("U", nu, N)
        S = ca.SX.sym("S", n_obstacles, N + 1)
        R = ca.SX.sym("R", 3, N + 1)
        D = ca.SX.sym("D", 2, N)
        P = ca.SX.sym("P", self.nx + nu)

        obj = 0
        g = []
        lbg = []
        ubg = []

        x0 = P[: self.nx]
        u_prev = P[self.nx :]

        w_y = 120.0
        w_psi = 20.0
        w_v = 10.0
        w_u = ca.diag(ca.DM([0.7, 3.5]))
        w_du = ca.diag(ca.DM([6.0, 45.0]))
        w_terminal = ca.diag(ca.DM([4.0, 60.0, 20.0, 14.0]))
        progress_reward = 7.0
        slack_linear_penalty = 120.0
        slack_quadratic_penalty = 4000.0
        road_slack_penalty = 2500.0
        rate_slack_penalty = 1800.0

        xk = x0

        for k in range(N):
            uk = U[:, k]
            y_ref = self._lane_reference_symbolic(xk[0])
            dy_dx = self._lane_reference_slope_symbolic(xk[0])
            psi_ref = ca.atan(dy_dx)
            v_ref = self._speed_reference_symbolic(xk[0])

            tracking_error = ca.vertcat(
                xk[0],
                xk[1] - y_ref,
                xk[2] - psi_ref,
                xk[3] - v_ref,
            )
            obj += (
                w_y * tracking_error[1] ** 2
                + w_psi * tracking_error[2] ** 2
                + w_v * tracking_error[3] ** 2
                + ca.mtimes([uk.T, w_u, uk])
                - progress_reward * xk[3]
            )

            du = uk - (u_prev if k == 0 else U[:, k - 1])
            obj += ca.mtimes([du.T, w_du, du])

            g.append(xk[1] - R[0, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.road_half_width)
            g.append(-xk[1] - R[0, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.road_half_width)

            g.append(du[0] - D[0, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.jerk_limit)
            g.append(-du[0] - D[0, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.jerk_limit)

            g.append(du[1] - D[1, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.steer_rate_limit)
            g.append(-du[1] - D[1, k])
            lbg.append(-ca.inf)
            ubg.append(self.cfg.steer_rate_limit)

            obj += road_slack_penalty * R[0, k] ** 2
            obj += rate_slack_penalty * D[0, k] ** 2 + rate_slack_penalty * D[1, k] ** 2

            for obs_index, obstacle in enumerate(self.obstacles):
                normalized_distance = ((xk[0] - obstacle.x) / obstacle.rx) ** 2 + (
                    (xk[1] - obstacle.y) / obstacle.ry
                ) ** 2
                g.append(normalized_distance + S[obs_index, k])
                lbg.append(1.0)
                ubg.append(ca.inf)
                obj += slack_linear_penalty * S[obs_index, k] + slack_quadratic_penalty * S[obs_index, k] ** 2

            g.append(xk[3])
            lbg.append(self.cfg.v_min)
            ubg.append(self.cfg.v_max)

            xk = self._rk4_step(xk, uk)

        y_ref_terminal = self._lane_reference_symbolic(xk[0])
        dy_dx_terminal = self._lane_reference_slope_symbolic(xk[0])
        psi_ref_terminal = ca.atan(dy_dx_terminal)
        v_ref_terminal = self._speed_reference_symbolic(xk[0])
        terminal_error = ca.vertcat(
            0.0,
            xk[1] - y_ref_terminal,
            xk[2] - psi_ref_terminal,
            xk[3] - v_ref_terminal,
        )
        obj += ca.mtimes([terminal_error.T, w_terminal, terminal_error])

        g.append(xk[1] - R[0, N])
        lbg.append(-ca.inf)
        ubg.append(self.cfg.road_half_width)
        g.append(-xk[1] - R[0, N])
        lbg.append(-ca.inf)
        ubg.append(self.cfg.road_half_width)
        obj += road_slack_penalty * R[0, N] ** 2
        g.append(xk[3])
        lbg.append(self.cfg.v_min)
        ubg.append(self.cfg.v_max)

        for obs_index, obstacle in enumerate(self.obstacles):
            normalized_distance = ((xk[0] - obstacle.x) / obstacle.rx) ** 2 + (
                (xk[1] - obstacle.y) / obstacle.ry
            ) ** 2
            g.append(normalized_distance + S[obs_index, N])
            lbg.append(1.0)
            ubg.append(ca.inf)
            obj += slack_linear_penalty * S[obs_index, N] + slack_quadratic_penalty * S[obs_index, N] ** 2

        decision_vars = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(S, -1, 1), ca.reshape(R, -1, 1), ca.reshape(D, -1, 1))

        lbx = []
        ubx = []
        for _ in range(N):
            lbx.extend([self.cfg.a_min, self.cfg.delta_min])
            ubx.extend([self.cfg.a_max, self.cfg.delta_max])
        for _ in range((N + 1) * n_obstacles):
            lbx.append(0.0)
            ubx.append(ca.inf)
        for _ in range(3 * (N + 1)):
            lbx.append(0.0)
            ubx.append(ca.inf)
        for _ in range(2 * N):
            lbx.append(0.0)
            ubx.append(ca.inf)

        nlp = {"x": decision_vars, "f": obj, "g": ca.vertcat(*g), "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 600,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.sb": "yes",
            "print_time": 0,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.constraint_function = ca.Function("constraint_function", [decision_vars, P], [ca.vertcat(*g)])
        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)

    def initial_guess(self, x_init):
        N = self.cfg.horizon
        guess_inputs = np.zeros((self.nu, N))
        guess_slacks = np.full((len(self.obstacles), N + 1), 2.0)
        guess_road_slacks = np.zeros((3, N + 1))
        guess_rate_slacks = np.zeros((2, N))

        return np.concatenate(
            [
                guess_inputs.reshape(-1, order="F"),
                guess_slacks.reshape(-1, order="F"),
                guess_road_slacks.reshape(-1, order="F"),
                guess_rate_slacks.reshape(-1, order="F"),
            ]
        )

    def shift_guess(self, solution):
        control_block = self.nu * self.cfg.horizon
        slack_block = len(self.obstacles) * (self.cfg.horizon + 1)
        road_block = 3 * (self.cfg.horizon + 1)
        rate_block = 2 * self.cfg.horizon
        controls = solution[:control_block].reshape(self.nu, self.cfg.horizon, order="F")
        slacks = solution[control_block : control_block + slack_block].reshape(
            len(self.obstacles), self.cfg.horizon + 1, order="F"
        )
        road_slacks = solution[control_block + slack_block : control_block + slack_block + road_block].reshape(
            3, self.cfg.horizon + 1, order="F"
        )
        rate_slacks = solution[
            control_block + slack_block + road_block : control_block + slack_block + road_block + rate_block
        ].reshape(2, self.cfg.horizon, order="F")

        shifted_controls = np.hstack([controls[:, 1:], controls[:, -1:]])
        shifted_slacks = np.hstack([slacks[:, 1:], slacks[:, -1:]])
        shifted_road = np.hstack([road_slacks[:, 1:], road_slacks[:, -1:]])
        shifted_rate = np.hstack([rate_slacks[:, 1:], rate_slacks[:, -1:]])
        return np.concatenate(
            [
                shifted_controls.reshape(-1, order="F"),
                shifted_slacks.reshape(-1, order="F"),
                shifted_road.reshape(-1, order="F"),
                shifted_rate.reshape(-1, order="F"),
            ]
        )

    def rollout(self, x, u):
        px, py, psi, v = x
        a, delta = u
        dt = self.cfg.dt
        l = self.cfg.wheelbase

        def f(state):
            xk, yk, psik, vk = state
            return np.array(
                [
                    vk * np.cos(psik),
                    vk * np.sin(psik),
                    vk * np.tan(delta) / l,
                    a,
                ]
            )

        k1 = f(np.array([px, py, psi, v]))
        k2 = f(np.array([px, py, psi, v]) + 0.5 * dt * k1)
        k3 = f(np.array([px, py, psi, v]) + 0.5 * dt * k2)
        k4 = f(np.array([px, py, psi, v]) + dt * k3)
        next_state = np.array([px, py, psi, v]) + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[3] = np.clip(next_state[3], self.cfg.v_min, self.cfg.v_max)
        next_state[1] = np.clip(next_state[1], -self.cfg.road_half_width, self.cfg.road_half_width)
        return next_state

    def solve(self, x_init, u_prev, guess):
        params = np.concatenate([x_init, u_prev])
        result = self.solver(
            x0=guess,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=params,
        )
        solution = np.array(result["x"]).flatten()
        status = self.solver.stats()["return_status"]
        g_values = np.array(self.constraint_function(solution, params)).flatten()
        lower_violation = np.maximum(self.lbg - g_values, 0.0)
        upper_violation = np.maximum(g_values - self.ubg, 0.0)
        max_violation = float(np.max(np.concatenate([lower_violation, upper_violation])))

        acceptable_status = (
            "Solve_Succeeded" in status
            or "Solved_To_Acceptable_Level" in status
            or ("Maximum_Iterations_Exceeded" in status and max_violation < 5.0e-2)
        )
        if not acceptable_status:
            raise RuntimeError(f"MPC solve failed with status: {status} and max constraint violation {max_violation:.3e}")

        control_block = self.nu * self.cfg.horizon
        controls = solution[:control_block].reshape(self.nu, self.cfg.horizon, order="F")
        predicted_states = self.rollout_sequence(x_init, controls)
        return solution, controls, predicted_states, float(result["f"])

    def rollout_sequence(self, x_init, controls):
        states = [np.array(x_init, dtype=float)]
        x = np.array(x_init, dtype=float)
        for k in range(controls.shape[1]):
            x = self.rollout(x, controls[:, k])
            states.append(x.copy())
        return np.array(states)


def simulate():
    cfg = MPCConfig()
    obstacles = obstacle_ellipses()
    controller = BicycleMPC(cfg, obstacles)

    x = np.array([0.0, 0.0, 0.0, 7.5])
    u_prev = np.array([0.0, 0.0])
    guess = controller.initial_guess(x)

    states = [x.copy()]
    controls = []
    costs = []
    solve_progress = []

    for step in range(cfg.steps):
        solution, predicted_controls, predicted_states, cost = controller.solve(x, u_prev, guess)
        u = predicted_controls[:, 0]
        x = controller.rollout(x, u)

        states.append(x.copy())
        controls.append(u.copy())
        costs.append(cost)
        solve_progress.append(predicted_states[-1, 0])

        u_prev = u
        guess = controller.shift_guess(solution)

        if x[0] > 52.0:
            break

    state_array = np.array(states)
    control_array = np.array(controls)
    time_state = cfg.dt * np.arange(state_array.shape[0])
    time_control = cfg.dt * np.arange(control_array.shape[0])

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    results = pd.DataFrame(
        {
            "time_s": time_control,
            "x_m": state_array[:-1, 0],
            "y_m": state_array[:-1, 1],
            "psi_rad": state_array[:-1, 2],
            "speed_mps": state_array[:-1, 3],
            "accel_mps2": control_array[:, 0],
            "steer_rad": control_array[:, 1],
            "stage_cost": costs,
            "predicted_terminal_x": solve_progress,
        }
    )
    results.to_csv(output_dir / "closed_loop_results.csv", index=False)

    plot_results(cfg, obstacles, state_array, control_array, time_state, time_control, costs, output_dir)


def plot_results(cfg, obstacles, state_array, control_array, time_state, time_control, costs, output_dir):
    x_dense = np.linspace(0.0, max(55.0, state_array[-1, 0] + 3.0), 500)
    y_dense = reference_lane(x_dense)
    v_dense = reference_speed(x_dense)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_path, ax_speed, ax_controls, ax_constraints = axes.flatten()

    ax_path.plot(x_dense, y_dense, "--", color="tab:blue", linewidth=2.0, label="reference centerline")
    ax_path.plot(state_array[:, 0], state_array[:, 1], color="tab:red", linewidth=2.6, label="closed-loop trajectory")
    ax_path.axhline(cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    ax_path.axhline(-cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    for obstacle in obstacles:
        theta = np.linspace(0.0, 2.0 * np.pi, 150)
        x_obs = obstacle.x + obstacle.rx * np.cos(theta)
        y_obs = obstacle.y + obstacle.ry * np.sin(theta)
        ax_path.fill(x_obs, y_obs, color="gray", alpha=0.35)
    ax_path.set_title("Vehicle Path and Obstacle Constraints")
    ax_path.set_xlabel("x position [m]")
    ax_path.set_ylabel("y position [m]")
    ax_path.legend(loc="upper right")
    ax_path.grid(True, alpha=0.25)

    ax_speed.plot(x_dense, v_dense, "--", color="tab:green", linewidth=2.0, label="speed reference")
    ax_speed.plot(state_array[:, 0], state_array[:, 3], color="tab:orange", linewidth=2.6, label="actual speed")
    ax_speed.axhline(cfg.v_min, color="black", linestyle=":", linewidth=1.0)
    ax_speed.axhline(cfg.v_max, color="black", linestyle=":", linewidth=1.0)
    ax_speed.set_title("Position-Dependent Speed Tracking")
    ax_speed.set_xlabel("x position [m]")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.legend(loc="upper right")
    ax_speed.grid(True, alpha=0.25)

    ax_controls.step(time_control, control_array[:, 0], where="post", linewidth=2.0, label="acceleration")
    axControls.step(time_control, control_array[:, 1], where="post", linewidth=2.0, label="steering")
    ax_controls.axhline(cfg.a_min, color="tab:blue", linestyle=":", linewidth=1.0)
    ax_controls.axhline(cfg.a_max, color="tab:blue", linestyle=":", linewidth=1.0)
    ax_controls.axhline(cfg.delta_min, color="tab:orange", linestyle=":", linewidth=1.0)
    ax_controls.axhline(cfg.delta_max, color="tab:orange", linestyle=":", linewidth=1.0)
    ax_controls.set_title("Constrained Control Inputs")
    ax_controls.set_xlabel("time [s]")
    ax_controls.set_ylabel("input value")
    ax_controls.legend(loc="upper right")
    ax_controls.grid(True, alpha=0.25)

    obstacle_margins = []
    for state in state_array[:-1]:
        margins = [
            ((state[0] - obs.x) / obs.rx) ** 2 + ((state[1] - obs.y) / obs.ry) ** 2 - 1.0
            for obs in obstacles
        ]
        obstacle_margins.append(min(margins))
    obstacle_margins = np.array(obstacle_margins)

    ax_constraints.plot(time_control, obstacle_margins, linewidth=2.2, label="min obstacle margin")
    ax_constraints.plot(time_control, costs, linewidth=2.0, label="stage objective")
    ax_constraints.axhline(0.0, color="black", linestyle=":", linewidth=1.0, label="active obstacle boundary")
    ax_constraints.set_title("Constraint Activity and Cost")
    ax_constraints.set_xlabel("time [s]")
    ax_constraints.set_ylabel("value")
    ax_constraints.legend(loc="upper right")
    ax_constraints.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "mpc_summary.png", dpi=180)
    fig.savefig(output_dir / "mpc_summary.svg")
    plt.close(fig)

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    y_ref_over_time = reference_lane(state_array[:, 0])
    v_ref_over_time = reference_speed(state_array[:, 0])
    axes2[0].plot(time_state, state_array[:, 1], linewidth=2.2, label="y")
    axes2[0].plot(time_state, y_ref_over_time, "--", linewidth=2.0, label="y reference")
    axes2[0].axhline(cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    axes2[0].axhline(-cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    axes2[0].set_ylabel("lateral position [m]")
    axes2[0].legend(loc="upper right")
    axes2[0].grid(True, alpha=0.25)

    axes2[1].plot(time_state, state_array[:, 2], linewidth=2.2, label="heading")
    axes2[1].plot(time_state, state_array[:, 3], linewidth=2.2, label="speed")
    axes2[1].plot(time_state, v_ref_over_time, "--", linewidth=2.0, label="speed ref")
    axes2[1].set_ylabel("heading / speed")
    axes2[1].legend(loc="upper right")
    axes2[1].grid(True, alpha=0.25)

    accel_rate = np.diff(np.concatenate([[0.0], control_array[:, 0]]))
    steer_rate = np.diff(np.concatenate([[0.0], control_array[:, 1]]))
    axes2[2].step(time_control, accel_rate, where="post", linewidth=2.0, label="delta accel")
    axes2[2].step(time_control, steer_rate, where="post", linewidth=2.0, label="delta steer")
    axes2[2].axhline(cfg.jerk_limit, color="tab:blue", linestyle=":", linewidth=1.0)
    axes2[2].axhline(-cfg.jerk_limit, color="tab:blue", linestyle=":", linewidth=1.0)
    axes2[2].axhline(cfg.steer_rate_limit, color="tab:orange", linestyle=":", linewidth=1.0)
    axes2[2].axhline(-cfg.steer_rate_limit, color="tab:orange", linestyle=":", linewidth=1.0)
    axes2[2].set_ylabel("input increments")
    axes2[2].set_xlabel("time [s]")
    axes2[2].legend(loc="upper right")
    axes2[2].grid(True, alpha=0.25)

    fig2.tight_layout()
    fig2.savefig(output_dir / "mpc_states.png", dpi=180)
    fig2.savefig(output_dir / "mpc_states.svg")
    plt.close(fig2)


if __name__ == "__main__":
    simulate()
