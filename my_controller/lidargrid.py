import numpy as np
import cv2
import math
from POSE import *                         # GetPose helper
import threading

# ─────────────────────────────────────────────────────────────────────────────
# Probability  ↔  log-odds helpers
# ─────────────────────────────────────────────────────────────────────────────
def probability_to_log_odds(probability: float) -> float:
    """Convert a probability in (0, 1) to log-odds."""
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("Probability must be in (0, 1).")
    return math.log(probability / (1.0 - probability))


def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Vectorised log-odds → probability."""
    return 1.0 - 1.0 / (1.0 + np.exp(log_odds))


# ─────────────────────────────────────────────────────────────────────────────
# LidarMap  ⇢  *mapping only* (no frontier code)
# ─────────────────────────────────────────────────────────────────────────────
class LidarOccupancyGridMapper:
    """Occupancy-grid mapping with a two-hit rule (no frontier extraction)."""

    # ------------------------------------------------------------------
    # Construction / configuration
    # ------------------------------------------------------------------
    def __init__(self, robot_node, pose_estimator, supervisor, lidar, time_step,
                 *, resolution: float = 0.01, grid_size: int = 1000):
        self.find_pose   = GetPose(robot_node)
        self.supervisor  = supervisor
        self.lidar       = lidar
        self.time_step   = time_step

        self.lock = threading.Lock()

        # Grid geometry
        self.GRID_SIZE   = int(grid_size)
        self.RESOLUTION  = float(resolution)             # [m] per pixel
        self.MAP_CENTER  = self.GRID_SIZE // 2

        # Log-odds grid (start at "unknown" ≡ 0.0)
        self.log_odds    = np.zeros((self.GRID_SIZE, self.GRID_SIZE),
                                    dtype=np.float32)
        self.grid_map    = np.full((self.GRID_SIZE, self.GRID_SIZE), 127,
                                   dtype=np.uint8)       # 0 = occ, 127 = unk, 255 = free

        # Bayesian constants
        self.l_occ   = probability_to_log_odds(0.9)      # hit bonus
        self.l_free  = probability_to_log_odds(0.30)     # miss penalty
        self.l_min, self.l_max = -10.0, 10.0
        self.l_occ_thresh = 4.5 * self.l_occ  # two-hit rule

        # Pose & bookkeeping
        self.origin        = None    # world frame origin (first call)
        self.current_pose  = None    # (x, y, yaw) in map frame
        self.scans_world   = []      # debug – raw hit points
        self.mapcarpose = (0,0)

        self.path = []  # for JPS pathfinding (not used in this class)
    
    def update_navigation_path(self, path):
        """Update the JPS path for external use."""
        self.path = path
    # ------------------------------------------------------------------
    # Integer Bresenham (helper)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_bresenham_line(x0, y0, x1, y1):
        """Return integer grid cells from (x0, y0) → (x1, y1)."""
        cells, dx, dy = [], abs(x1 - x0), abs(y1 - y0)
        x, y, sx, sy = x0, y0, (-1 if x0 > x1 else 1), (-1 if y0 > y1 else 1)
        if dx > dy:
            err = dx // 2
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy // 2
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return cells

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def initialize_grid_origin(self):
        self.origin = self.find_pose.get_initial_pose()
        return self.origin

    def set_current_global_pose(self, dx_world, dy_world, yaw):
        """(dx, dy, yaw) already expressed in *map* metres/radians."""
        self.current_pose = (dx_world, dy_world, yaw)

    # ------------------------------------------------------------------
    # Map update  (no frontier overlay)
    # ------------------------------------------------------------------
    def integrate_lidar_scan(self, ranges, angular_res, min_angle):
        """Fuse one Lidar scan into the log-odds grid (v2: proper ray carving)."""
        if self.current_pose is None:
            return
        x_map, y_map, yaw_rel = self.current_pose

        r = np.asarray(ranges, dtype=np.float32)
        if r.size == 0:
            return

        # Webots returns maxRange when no hit; we still want to carve FREE space.
        min_r = float(self.lidar.getMinRange())
        max_r = float(self.lidar.getMaxRange())

        # Choose beams that are finite and above min range
        finite = np.isfinite(r)
        valid = finite & (r > min_r)
        if not np.any(valid):
            return

        idx = np.nonzero(valid)[0].astype(np.float32)
        r_valid = r[valid]

        # Downsample beams slightly for speed (tune if needed)
        beam_skip = 2
        if idx.size > 200:
            idx = idx[::beam_skip]
            r_valid = r_valid[::beam_skip]

        beam_ang = min_angle + idx * -angular_res
        world_ang = yaw_rel + beam_ang

        # Robot grid coords
        gx0 = int(round(x_map / self.RESOLUTION)) + self.MAP_CENTER
        gy0 = self.MAP_CENTER - int(round(y_map / self.RESOLUTION))

        # For each beam: carve FREE up to end, and mark OCC only if it is a real hit (< max_r - eps)
        eps = 1e-3
        for rr, ang in zip(r_valid, world_ang):
            rr = float(rr)

            hit = rr < (max_r - eps)

            # If no hit, carve to max_r (or capped)
            rr_carve = rr if hit else max_r
            x_end = x_map + rr_carve * math.cos(ang)
            y_end = y_map + rr_carve * math.sin(ang)

            ex = int(round(x_end / self.RESOLUTION)) + self.MAP_CENTER
            ey = self.MAP_CENTER - int(round(y_end / self.RESOLUTION))

            # bounds check
            if ex < 0 or ex >= self.GRID_SIZE or ey < 0 or ey >= self.GRID_SIZE:
                continue

            # FREE along ray (excluding endpoint)
            for fx, fy in self.compute_bresenham_line(gx0, gy0, ex, ey)[:-1]:
                if 0 <= fx < self.GRID_SIZE and 0 <= fy < self.GRID_SIZE:
                    self.log_odds[fy, fx] += self.l_free

            # OCC at endpoint only on real hits
            if hit:
                self.log_odds[ey, ex] += self.l_occ

        # Clamp
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

        # Render discrete occupancy grid
        self.grid_map.fill(127)
        self.grid_map[self.log_odds >= self.l_occ_thresh] = 0
        self.grid_map[self.log_odds <= self.l_free]       = 255

    # ------------------------------------------------------------------
    # Robot footprint (re-used by frontier extractor)
    # ------------------------------------------------------------------
    def get_robot_footprint(self):
        """Return (length_px, width_px) footprint in *grid* pixels."""
        length_m, width_m = 0.21, 0.245
        return int(length_m / self.RESOLUTION), int(width_m / self.RESOLUTION)

    # ------------------------------------------------------------------
    # Internal helper used by FrontierExtractor for visualisation
    # ------------------------------------------------------------------
    def draw_robot_on_grid(self, img):
        if self.current_pose is None:
            return

        occ_img = cv2.cvtColor(self.grid_map, cv2.COLOR_GRAY2BGR)

        # Draw robot position
        x_map, y_map, _ = self.current_pose
        gx = int(round(x_map / self.RESOLUTION)) + self.MAP_CENTER
        gy = self.MAP_CENTER - int(round(y_map / self.RESOLUTION))
        if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
            cv2.circle(occ_img, (gx, gy), 5, (0, 0, 255), -1)

        cv2.imshow("Occupancy Grid", occ_img)
        cv2.waitKey(1)  # required for OpenCV display loop