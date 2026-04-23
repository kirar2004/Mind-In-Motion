import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig-casadi-mpc"))

import casadi as ca
import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
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
class Scenario:
    name: str
    label: str
    obstacles: tuple[Obstacle, Obstacle]


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


def make_obstacles(
    first_position=(20.0, 0.0),
    second_position=(36.0, 2.0),
    first_radii=(3.0, 1.3),
    second_radii=(2.8, 1.25),
):
    """Build obstacle ellipses from center and radius parameters."""
    return (
        Obstacle(x=first_position[0], y=first_position[1], rx=first_radii[0], ry=first_radii[1]),
        Obstacle(x=second_position[0], y=second_position[1], rx=second_radii[0], ry=second_radii[1]),
    )


def scenario_library():
    return [
        Scenario(
            name="baseline",
            label="Baseline: staggered obstacles",
            obstacles=make_obstacles((20.0, 0.0), (36.0, 2.0)),
        ),
        Scenario(
            name="lower_upper",
            label="Lower then upper obstacles",
            obstacles=make_obstacles((17.0, -1.5), (34.0, 2.3)),
        ),
        Scenario(
            name="middle_gate",
            label="Middle gate",
            obstacles=make_obstacles((20.0, -1.8), (35.0, 1.8)),
        ),
        Scenario(
            name="late_gate",
            label="Late gate",
            obstacles=make_obstacles((23.0, -1.4), (39.0, 1.4)),
        ),
        Scenario(
            name="same_lower_side",
            label="Same-side lower obstacles",
            obstacles=make_obstacles((20.0, -1.6), (36.0, -0.4)),
        ),
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


def simulate_scenario(cfg, scenario):
    obstacles = scenario.obstacles
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

    return {
        "scenario": scenario,
        "cfg": cfg,
        "obstacles": obstacles,
        "states": state_array,
        "controls": control_array,
        "time_state": time_state,
        "time_control": time_control,
        "costs": np.array(costs),
        "solve_progress": np.array(solve_progress),
    }


def scenario_metrics(result):
    cfg = result["cfg"]
    obstacles = result["obstacles"]
    states = result["states"]
    controls = result["controls"]
    time_control = result["time_control"]
    margins = obstacle_margin_series(obstacles, states[:-1])
    y_ref = reference_lane(states[:, 0])
    v_ref = reference_speed(states[:, 0])
    if len(time_control) > 1:
        accel_variation = np.sum(np.abs(np.diff(controls[:, 0])))
        steer_variation = np.sum(np.abs(np.diff(controls[:, 1])))
    else:
        accel_variation = 0.0
        steer_variation = 0.0
    return {
        "scenario": result["scenario"].name,
        "label": result["scenario"].label,
        "obstacle_1_x": obstacles[0].x,
        "obstacle_1_y": obstacles[0].y,
        "obstacle_2_x": obstacles[1].x,
        "obstacle_2_y": obstacles[1].y,
        "final_x_m": states[-1, 0],
        "max_abs_y_m": np.max(np.abs(states[:, 1])),
        "rms_lateral_error_m": np.sqrt(np.mean((states[:, 1] - y_ref) ** 2)),
        "rms_speed_error_mps": np.sqrt(np.mean((states[:, 3] - v_ref) ** 2)),
        "max_speed_mps": np.max(states[:, 3]),
        "max_abs_steer_rad": np.max(np.abs(controls[:, 1])),
        "min_obstacle_margin": np.min(margins),
        "accel_variation": accel_variation,
        "steer_variation": steer_variation,
        "mean_stage_cost": np.mean(result["costs"]),
        "elapsed_time_s": cfg.dt * max(len(states) - 1, 0),
    }


def save_primary_results(result, code_dir):
    state_array = result["states"]
    control_array = result["controls"]
    time_control = result["time_control"]

    results = pd.DataFrame(
        {
            "time_s": time_control,
            "x_m": state_array[:-1, 0],
            "y_m": state_array[:-1, 1],
            "psi_rad": state_array[:-1, 2],
            "speed_mps": state_array[:-1, 3],
            "accel_mps2": control_array[:, 0],
            "steer_rad": control_array[:, 1],
            "stage_cost": result["costs"],
            "predicted_terminal_x": result["solve_progress"],
        }
    )
    results.to_csv(code_dir / "closed_loop_results.csv", index=False)


def simulate():
    cfg = MPCConfig()
    plots_dir = PROJECT_DIR.parent / "plots"
    code_dir = PROJECT_DIR
    plots_dir.mkdir(exist_ok=True)

    scenarios = scenario_library()
    primary = simulate_scenario(cfg, scenarios[0])
    comparison_results = [simulate_scenario(cfg, scenario) for scenario in scenarios[1:]]

    save_primary_results(primary, code_dir)
    pd.DataFrame([scenario_metrics(result) for result in [primary] + comparison_results]).to_csv(
        code_dir / "scenario_comparison_metrics.csv",
        index=False,
    )

    plot_geometry(cfg, primary, plots_dir)
    plot_results(primary, plots_dir)
    animate_closed_loop(primary, plots_dir)
    plot_scenario_comparison(comparison_results, plots_dir)


def obstacle_margin_series(obstacles, states):
    obstacle_margins = []
    for state in states:
        margins = [
            ((state[0] - obs.x) / obs.rx) ** 2 + ((state[1] - obs.y) / obs.ry) ** 2 - 1.0
            for obs in obstacles
        ]
        obstacle_margins.append(min(margins))
    return np.array(obstacle_margins)


def obstacle_patch(ax, obstacle, color="0.45", alpha=0.35, label=None):
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    x_obs = obstacle.x + obstacle.rx * np.cos(theta)
    y_obs = obstacle.y + obstacle.ry * np.sin(theta)
    ax.fill(x_obs, y_obs, color=color, alpha=alpha, label=label)
    ax.plot(x_obs, y_obs, color=color, linewidth=1.0, alpha=0.8)


def draw_bicycle(ax, x, y, psi, length=2.8, width=1.25, color="#dc2626", alpha=0.95):
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    rotation = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    body = corners @ rotation.T + np.array([x, y])
    patch = Polygon(body, closed=True, facecolor=color, edgecolor="#7f1d1d", alpha=alpha, linewidth=1.0)
    ax.add_patch(patch)

    front_center = np.array([0.65 * half_l, 0.0]) @ rotation.T + np.array([x, y])
    rear_center = np.array([-0.65 * half_l, 0.0]) @ rotation.T + np.array([x, y])
    front = Circle(front_center, radius=0.16, facecolor="#111827", edgecolor="white", linewidth=0.6, zorder=5)
    rear = Circle(rear_center, radius=0.16, facecolor="#111827", edgecolor="white", linewidth=0.6, zorder=5)
    ax.add_patch(front)
    ax.add_patch(rear)
    return [patch, front, rear]


def format_path_axis(ax, cfg, x_dense, y_dense):
    ax.plot(x_dense, y_dense, "--", color="tab:blue", linewidth=2.0, label="reference centerline")
    ax.axhline(cfg.road_half_width, color="black", linestyle=":", linewidth=1.0, label="road boundary")
    ax.axhline(-cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    ax.fill_between(x_dense, -cfg.road_half_width, cfg.road_half_width, color="#e5e7eb", alpha=0.28)
    ax.set_xlim(0.0, max(55.0, x_dense[-1]))
    ax.set_ylim(-5.2, 5.2)
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.grid(True, alpha=0.25)


def plot_geometry(cfg, result, output_dir):
    obstacles = result["obstacles"]
    state_array = result["states"]
    x_dense = np.linspace(0.0, 55.0, 600)
    y_dense = reference_lane(x_dense)

    fig, ax = plt.subplots(figsize=(13, 5.4))
    format_path_axis(ax, cfg, x_dense, y_dense)
    ax.plot(state_array[:, 0], state_array[:, 1], color="#dc2626", linewidth=2.3, label="closed-loop trajectory")
    for index, obstacle in enumerate(obstacles, start=1):
        obstacle_patch(ax, obstacle, label=f"obstacle {index}")
        ax.annotate(
            f"O{index}=({obstacle.x:.1f}, {obstacle.y:.1f})\n$r_x$={obstacle.rx:.1f}, $r_y$={obstacle.ry:.1f}",
            xy=(obstacle.x, obstacle.y),
            xytext=(obstacle.x - 5.6, obstacle.y + 2.4),
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.0},
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
        )

    for sample in [8, 18, 30, 40]:
        if sample < len(state_array):
            draw_bicycle(ax, state_array[sample, 0], state_array[sample, 1], state_array[sample, 2], alpha=0.42)
    ax.set_title("Scenario Geometry: Lane, Parameterized Obstacles, and Bicycle States")
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "mpc-geometry.png", dpi=180)
    fig.savefig(output_dir / "mpc-geometry.svg")
    plt.close(fig)


def plot_results(result, output_dir):
    cfg = result["cfg"]
    obstacles = result["obstacles"]
    state_array = result["states"]
    control_array = result["controls"]
    time_state = result["time_state"]
    time_control = result["time_control"]
    costs = result["costs"]

    x_dense = np.linspace(0.0, max(55.0, state_array[-1, 0] + 3.0), 500)
    y_dense = reference_lane(x_dense)
    v_dense = reference_speed(x_dense)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False, gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax_path, ax_speed = axes

    format_path_axis(ax_path, cfg, x_dense, y_dense)
    ax_path.plot(state_array[:, 0], state_array[:, 1], color="tab:red", linewidth=2.6, label="closed-loop trajectory")
    for index, obstacle in enumerate(obstacles, start=1):
        obstacle_patch(ax_path, obstacle, label=f"obstacle {index}")
    ax_path.set_title("Closed-Loop Path and Obstacle Constraints")
    ax_path.legend(loc="upper right", ncols=2)

    ax_speed.plot(x_dense, v_dense, "--", color="tab:green", linewidth=2.0, label="speed reference along path")
    ax_speed.plot(state_array[:, 0], state_array[:, 3], color="tab:orange", linewidth=2.6, label="closed-loop speed")
    ax_speed.axhline(cfg.v_min, color="black", linestyle=":", linewidth=1.0)
    ax_speed.axhline(cfg.v_max, color="black", linestyle=":", linewidth=1.0)
    ax_speed.fill_between(x_dense, cfg.v_min, cfg.v_max, color="#dcfce7", alpha=0.25, label="speed limits")
    ax_speed.set_title("Velocity Profile")
    ax_speed.set_xlabel("x position [m]")
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.legend(loc="upper right")
    ax_speed.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "mpc-summary.png", dpi=180)
    fig.savefig(output_dir / "mpc-summary.svg")
    plt.close(fig)

    fig_inputs, axes_inputs = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    y_ref_over_time = reference_lane(state_array[:, 0])
    v_ref_over_time = reference_speed(state_array[:, 0])
    axes_inputs[0].plot(time_state, state_array[:, 1], linewidth=2.2, label="y")
    axes_inputs[0].plot(time_state, y_ref_over_time, "--", linewidth=2.0, label="y reference")
    axes_inputs[0].axhline(cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    axes_inputs[0].axhline(-cfg.road_half_width, color="black", linestyle=":", linewidth=1.0)
    axes_inputs[0].set_ylabel("lateral position [m]")
    axes_inputs[0].legend(loc="upper right")
    axes_inputs[0].grid(True, alpha=0.25)

    axes_inputs[1].plot(time_state, state_array[:, 2], linewidth=2.2, label="heading")
    axes_inputs[1].plot(time_state, state_array[:, 3], linewidth=2.2, label="speed")
    axes_inputs[1].plot(time_state, v_ref_over_time, "--", linewidth=2.0, label="speed ref")
    axes_inputs[1].set_ylabel("heading / speed")
    axes_inputs[1].legend(loc="upper right")
    axes_inputs[1].grid(True, alpha=0.25)

    accel_rate = np.diff(np.concatenate([[0.0], control_array[:, 0]]))
    steer_rate = np.diff(np.concatenate([[0.0], control_array[:, 1]]))
    axes_inputs[2].step(time_control, accel_rate, where="post", linewidth=2.0, label="delta accel")
    axes_inputs[2].step(time_control, steer_rate, where="post", linewidth=2.0, label="delta steer")
    axes_inputs[2].axhline(cfg.jerk_limit, color="tab:blue", linestyle=":", linewidth=1.0)
    axes_inputs[2].axhline(-cfg.jerk_limit, color="tab:blue", linestyle=":", linewidth=1.0)
    axes_inputs[2].axhline(cfg.steer_rate_limit, color="tab:orange", linestyle=":", linewidth=1.0)
    axes_inputs[2].axhline(-cfg.steer_rate_limit, color="tab:orange", linestyle=":", linewidth=1.0)
    axes_inputs[2].set_ylabel("input increments")
    axes_inputs[2].set_xlabel("time [s]")
    axes_inputs[2].legend(loc="upper right")
    axes_inputs[2].grid(True, alpha=0.25)

    fig_inputs.tight_layout()
    fig_inputs.savefig(output_dir / "mpc-states.png", dpi=180)
    fig_inputs.savefig(output_dir / "mpc-states.svg")
    plt.close(fig_inputs)


def animate_closed_loop(result, output_dir):
    cfg = result["cfg"]
    obstacles = result["obstacles"]
    states = result["states"]
    x_dense = np.linspace(0.0, max(55.0, states[-1, 0] + 3.0), 500)
    y_dense = reference_lane(x_dense)
    frame_indices = np.unique(np.linspace(0, len(states) - 1, min(52, len(states)), dtype=int))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))

    def draw_frame(frame_index):
        ax.clear()
        format_path_axis(ax, cfg, x_dense, y_dense)
        for index, obstacle in enumerate(obstacles, start=1):
            obstacle_patch(ax, obstacle, label=f"obstacle {index}")
        state_index = frame_indices[frame_index]
        ax.plot(states[: state_index + 1, 0], states[: state_index + 1, 1], color="#dc2626", linewidth=2.5)
        draw_bicycle(ax, states[state_index, 0], states[state_index, 1], states[state_index, 2])
        ax.set_title(f"Animated Closed-Loop Motion, t = {cfg.dt * state_index:.1f} s")
        ax.legend(loc="upper right", ncols=2)

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=130, repeat=True)
    anim.save(output_dir / "mpc-animation.gif", writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def plot_scenario_comparison(results, output_dir):
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    labels = [result["scenario"].label for result in results]

    fig_paths, axes_paths = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, result, color in zip(axes_paths.ravel(), results, colors):
        cfg = result["cfg"]
        states = result["states"]
        x_dense = np.linspace(0.0, max(55.0, states[-1, 0] + 3.0), 500)
        y_dense = reference_lane(x_dense)
        format_path_axis(ax, cfg, x_dense, y_dense)
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2.4, label="closed-loop trajectory")
        for index, obstacle in enumerate(result["obstacles"], start=1):
            obstacle_patch(ax, obstacle, label=f"obstacle {index}")
        ax.set_title(result["scenario"].label)
        ax.legend(loc="upper right", fontsize=8)
    fig_paths.suptitle("Four Obstacle-Position Scenarios", y=0.995)
    fig_paths.tight_layout()
    fig_paths.savefig(output_dir / "mpc-scenario-comparison.png", dpi=180)
    fig_paths.savefig(output_dir / "mpc-scenario-comparison.svg")
    plt.close(fig_paths)

    fig_controls, axes_controls = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for result, color, label in zip(results, colors, labels):
        controls = result["controls"]
        time_control = result["time_control"]
        axes_controls[0].step(time_control, controls[:, 0], where="post", color=color, linewidth=2.0, label=label)
        axes_controls[1].step(time_control, controls[:, 1], where="post", color=color, linewidth=2.0, label=label)
    cfg = results[0]["cfg"]
    axes_controls[0].axhline(cfg.a_min, color="black", linestyle=":", linewidth=1.0)
    axes_controls[0].axhline(cfg.a_max, color="black", linestyle=":", linewidth=1.0)
    axes_controls[0].set_ylabel("acceleration [m/s^2]")
    axes_controls[0].set_title("Acceleration Profiles")
    axes_controls[0].grid(True, alpha=0.25)
    axes_controls[0].legend(loc="upper right", fontsize=8)
    axes_controls[1].axhline(cfg.delta_min, color="black", linestyle=":", linewidth=1.0)
    axes_controls[1].axhline(cfg.delta_max, color="black", linestyle=":", linewidth=1.0)
    axes_controls[1].set_ylabel("steering [rad]")
    axes_controls[1].set_xlabel("time [s]")
    axes_controls[1].set_title("Steering Profiles")
    axes_controls[1].grid(True, alpha=0.25)
    fig_controls.tight_layout()
    fig_controls.savefig(output_dir / "mpc-control-comparison.png", dpi=180)
    fig_controls.savefig(output_dir / "mpc-control-comparison.svg")
    plt.close(fig_controls)

    metrics = pd.DataFrame([scenario_metrics(result) for result in results])
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(13, 4.4))
    metric_specs = [
        ("min_obstacle_margin", "minimum obstacle margin"),
        ("rms_speed_error_mps", "RMS speed error [m/s]"),
        ("steer_variation", "total steering variation"),
    ]
    x = np.arange(len(metrics))
    short_labels = [result["scenario"].name.replace("_", "\n") for result in results]
    for ax, (column, title) in zip(axes_metrics, metric_specs):
        ax.bar(x, metrics[column], color=colors, alpha=0.86)
        ax.set_xticks(x, short_labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig_metrics.tight_layout()
    fig_metrics.savefig(output_dir / "mpc-metrics-comparison.png", dpi=180)
    fig_metrics.savefig(output_dir / "mpc-metrics-comparison.svg")
    plt.close(fig_metrics)


if __name__ == "__main__":
    simulate()
