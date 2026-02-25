# dwa.py  (UPDATED: obstacle-aware DWA using LiDAR point obstacles)

import math
import numpy as np


class DynamicWindowPlanner:
    def __init__(self, config):
        self.cfg = config

    def compute_motion(self, state, velocity, angular_velocity, time_step):
        x, y, theta = state
        x += velocity * math.cos(theta) * time_step
        y += velocity * math.sin(theta) * time_step
        theta += angular_velocity * time_step
        return x, y, theta

    def calculate_dynamic_window(self, current_velocity, current_angular_velocity):
        velocity_space = [self.cfg['min_speed'], self.cfg['max_speed'],
                          -self.cfg['max_omega'], self.cfg['max_omega']]

        dynamic_window = [
            max(velocity_space[0], current_velocity - self.cfg['acc'] * self.cfg['dt']),
            min(velocity_space[1], current_velocity + self.cfg['acc'] * self.cfg['dt']),
            max(velocity_space[2], current_angular_velocity - self.cfg['omega_acc'] * self.cfg['dt']),
            min(velocity_space[3], current_angular_velocity + self.cfg['omega_acc'] * self.cfg['dt'])
        ]
        return dynamic_window

    def predict_trajectory(self, state, velocity, angular_velocity):
        trajectory = [state]
        time = 0.0
        while time <= self.cfg['predict_time']:
            state = self.compute_motion(state, velocity, angular_velocity, self.cfg['dt'])
            trajectory.append(state)
            time += self.cfg['dt']
        return trajectory

    def evaluate_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1][0]
        dy = goal[1] - trajectory[-1][1]
        return math.hypot(dx, dy)

    @staticmethod
    def calculate_min_distance(px, py, obstacles):
        # obstacles: (N,2)
        if obstacles is None or len(obstacles) == 0:
            return float("inf")
        dx = obstacles[:, 0] - px
        dy = obstacles[:, 1] - py
        return float(np.min(dx * dx + dy * dy)) ** 0.5

    def evaluate_obstacle_cost(self, trajectory, obstacles):
        """
        Hard collision check + smooth clearance cost.
        - If any point in trajectory comes closer than robot_radius -> INF (reject)
        - Otherwise: cost proportional to 1/min_clearance
        """
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        robot_radius = self.cfg.get("robot_radius", 0.18)  # meters (tune 0.16-0.22)
        min_distance = float("inf")
        for (x, y, _) in trajectory:
            distance = self.calculate_min_distance(x, y, obstacles)
            if distance < robot_radius:
                return float("inf")  # collision
            if distance < min_distance:
                min_distance = distance

        # encourage clearance (bigger distance => smaller cost)
        # clamp to avoid huge values when min_distance is very small but > radius
        min_distance = max(min_distance, robot_radius + 1e-3)
        return 1.0 / min_distance

    def plan_dwa(self, state, goal, current_velocity, current_angular_velocity, obstacles=None):
        dynamic_window = self.calculate_dynamic_window(current_velocity, current_angular_velocity)

        best_score = float('inf')
        best_control = [0.0, 0.0]
        best_trajectory = []

        velocity_samples = self.cfg.get("v_samples", 7)
        angular_velocity_samples = self.cfg.get("w_samples", 9)

        for velocity in np.linspace(dynamic_window[0], dynamic_window[1], velocity_samples):
            for angular_velocity in np.linspace(dynamic_window[2], dynamic_window[3], angular_velocity_samples):
                trajectory = self.predict_trajectory(state, velocity, angular_velocity)

                goal_cost = self.evaluate_goal_cost(trajectory, goal)
                speed_cost = (self.cfg['max_speed'] - velocity)

                obstacle_cost = self.evaluate_obstacle_cost(trajectory, obstacles)
                if math.isinf(obstacle_cost):
                    continue

                total_cost = (
                    self.cfg['goal_cost_gain'] * goal_cost +
                    self.cfg['speed_cost_gain'] * speed_cost +
                    self.cfg.get('obstacle_cost_gain', 1.0) * obstacle_cost
                )

                if total_cost < best_score:
                    best_score = total_cost
                    best_control = [velocity, angular_velocity]
                    best_trajectory = trajectory

        return best_control, best_trajectory

    def compute_wheel_speeds(self, velocity, angular_velocity):
        wheel_base = self.cfg['wheel_base']
        left_wheel = velocity - angular_velocity * wheel_base / 2
        right_wheel = velocity + angular_velocity * wheel_base / 2
        return left_wheel, right_wheel
