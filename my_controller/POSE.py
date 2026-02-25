import math

class PoseEstimator:
    """
    Computes the robot's 2D pose in the X-Y plane (ignoring Z vertical),
    relative to a fixed world frame. Axes stay aligned with the world axes.
    Outputs yaw (deg), dx_rel, dy_rel rounded to 4 decimals.
    """
    def __init__(self, robot_node):
        self.robot_node = robot_node

        # Save initial absolute position (x, y)
        x0, y0, _ = self.robot_node.getPosition()
        self.initial_x = x0
        self.initial_y = y0

        # Save initial absolute yaw
        orientation = self.robot_node.getOrientation()
        initial_yaw_rad = self._calculate_yaw(orientation)
        self.initial_yaw = initial_yaw_rad

    def _calculate_yaw(self, rotation_matrix):
        """
        Extract yaw (heading around world-Z) from the flat 3×3 matrix R
        returned by Webots: [r00, r01, r02, r10, r11, r12, r20, r21, r22].
        yaw θ satisfies r10 = sinθ, r00 = cosθ.
        """
        return math.atan2(rotation_matrix[3], rotation_matrix[0])

    def get_initial_pose(self):
        return self.initial_x, self.initial_y, self.initial_yaw

    def compute_relative_pose(self):
        """
        Returns:
          yaw (deg): heading in world frame, normalized to [-180, 180)
          dx_rel, dy_rel (m): translation along world X and Y axes,
                              relative to start; each rounded to 4 decimals
        """
        # 1) Current absolute pose (x, y)
        x, y, _ = self.robot_node.getPosition()
        orientation = self.robot_node.getOrientation()

        self.initial_x, self.initial_y, self.initial_yaw = self.get_initial_pose()
        # 2) Absolute yaw
        current_yaw = self._calculate_yaw(orientation)
        # 3) Translation in world frame since start
        dx_world = x - self.initial_x
        dy_world = y - self.initial_y
        # 4) Round
        return round(current_yaw, 4), round(dx_world, 4), round(dy_world, 4)
