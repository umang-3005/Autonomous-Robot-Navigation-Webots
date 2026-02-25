# rosbor_python.py



"""
Complete Fully Automated Semantic Navigation System
Mission: START → BLUE pillar → YELLOW pillar


- LiDAR-camera fusion for precise object localization
- Multi-frame confirmation (no false positives)
- Detects: BLUE pillars, YELLOW pillars, GREEN poison, RED walls
- Automatic waypoint extraction and path planning
- Cost-aware planning (avoid green, penalize red)
- Real-time semantic map visualization
- automatic replanning when path invalid / semantics change
- frontier-based autonomous exploration mode
"""



from controller import Supervisor, Keyboard
import numpy as np
import cv2
import math
import threading
import time

# --- Pygame (for visualization; replaces cv2.imshow windows) ---
try:
    import pygame
    _PYGAME_OK = True
except Exception:
    pygame = None
    _PYGAME_OK = False


from lidargrid import LidarMap
from frontier import FrontierExtractor
from POSE import GetPose
from camera_semantic_detector import CameraSemanticDetector
from cost_aware_planner import CostAwarePlanner
from dwa import DynamicWindowApproach



# ============================================================================
# CONFIGURATION
# ============================================================================



TIME_STEP = 32
MAX_SPEED = 10
BASE_SPEED = 4
ROT_THRESHOLD = math.radians(1.0)

BLUE_CONFIRM_DIST   = 0.65   # confirm only if robot is within this distance (meters)
YELLOW_CONFIRM_DIST = 0.65

# Camera detection settings
CAMERA_DETECTION_INTERVAL = 0.05  # 20 FPS


# Navigation parameters
WAYPOINT_REACHED_THRESHOLD = 0.15  # meters
GOAL_APPROACH_OFFSET = 0.25        # offset from pillar surface


# Replanning parameters
REPLAN_INTERVAL = 2.0              # minimum seconds between global replans
PATH_CHECK_STEP = 1               # check every 4th path cell for obstacles             was 4
PATH_LOOKAHEAD = 40                # max cells ahead to check for validity


# Frontier exploration parameters
FRONTIER_MIN_SIZE = 50
FRONTIER_GOAL_TOL = 0.15
FRONTIER_MAX_FAILED = 5
FRONTIER_MIN_COUNT_FOR_EXPLORATION = 5


# Initial scan duration (seconds) before robot starts to move
INITIAL_SCAN_DURATION = 2.0


# Local exploration safety parameters (kept for compatibility)
LOCAL_LOOKAHEAD = 0.6
LOCAL_MIN_FREE_DIST = 0.3
LOCAL_LATERAL_CLEAR = 0.25



# ============================================================================
# MAIN CONTROLLER CLASS
# ============================================================================



class RosbotAutonomousNavigator:
    """Fully autonomous START→BLUE→YELLOW navigation with semantic awareness"""


    KEY_W, KEY_S, KEY_A, KEY_D, KEY_SPACE, KEY_E = map(ord, "WSAD E")


    def __init__(self):

        # Always-defined members (so early init failures don't crash cleanup)
        self.initialized = False
        self.map_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prev_yaw = None

        # --- Anti-spin supervisor ---
        self.spin_start_time = None
        self.spin_yaw_ref = None
        self.spin_trans_ref = None
        self.escape_until = 0.0
        self.escape_phase = 0          # 0 reverse, 1 rotate, 2 forward
        self.escape_dir = 0            # +1 left, -1 right
        self.blue_pause_until = 0.0
        self.yellow_close_since = None
        self.blue_close_since = None
        self.first_goal = None
        self.second_goal = None



        """Initialize all components"""
        print("\n" + "="*60)
        print("🤖 RosBot Fully Automated Semantic Navig ation")
        print("🎯 Mission: START → BLUE → YELLOW")
        print("🟢 Detects: BLUE, YELLOW, GREEN (poison), RED (walls)")
        print("="*60 + "\n")


        # Webots supervisor and keyboard
        self.supervisor = Supervisor()
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)


        # Robot node
        robot_def_name = "rosbot"
        self.robot_node = self.supervisor.getFromDef(robot_def_name)
        if self.robot_node is None:
            print(f"❌ FATAL: Robot DEF '{robot_def_name}' not found!")
            return
        print(f"✓ Robot '{robot_def_name}' initialized")


        # Pose estimator
        self.pose = GetPose(self.robot_node)

        # LiDAR sensor (robust name lookup)
        self.lidar = None
        self.lidar_name = None
        for _name in ("laser", "lidar", "LDS-01", "LDS_01", "lds-01", "hokuyo", "Hokuyo"):
            try:
                dev = self.supervisor.getDevice(_name)
            except Exception:
                dev = None
            if dev is not None:
                self.lidar = dev
                self.lidar_name = _name
                break

        if self.lidar is None:
            print("❌ FATAL: LiDAR device not found (tried: laser, lidar, LDS-01, LDS_01, lds-01, hokuyo, Hokuyo)")
            return
        self.lidar.enable(TIME_STEP)
        self.lidar.disablePointCloud()
        print(f"✓ LiDAR initialized (name: {self.lidar_name})")



        # Camera
        print("\n📹 Initializing RGB camera...")
        self.camera = None
        try:
            self.camera = self.supervisor.getDevice("camera rgb")
            if self.camera:
                self.camera.enable(TIME_STEP)
                width = self.camera.getWidth()
                height = self.camera.getHeight()
                fov_deg = math.degrees(self.camera.getFov())
                print("✅ Camera initialized")
                print(f"  Resolution: {width} x {height}")
                print(f"  FOV: {fov_deg:.1f}°")
            else:
                print("⚠️ Camera not found")
        except Exception as e:
            print(f"⚠️ Camera error: {e}")
            self.camera = None


        # Mapping system
        self.grid = LidarMap(
            self.robot_node,
            self.pose,
            self.supervisor,
            self.lidar,
            TIME_STEP,
            resolution=0.02,
            grid_size=500
        )


        # Frontier extractor (for exploration algorithms)
        self.frontier = FrontierExtractor(self.grid, min_frontier_cells=FRONTIER_MIN_SIZE)
        print("✓ Occupancy grid mapping initialized")


        # Motors
        motor_names = ["fl_wheel_joint", "fr_wheel_joint",
                       "rl_wheel_joint", "rr_wheel_joint"]
        self.motors = [self.supervisor.getDevice(name) for name in motor_names]
        if any(m is None for m in self.motors):
            print("❌ FATAL: Motors not found!")
            return
        for motor in self.motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        print("✓ Motors initialized")


        # Shared state
        self.map_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prev_yaw = None
        self.blue_seen = False
        self.yellow_seen = False


        # State machine
        self.state = "INITIAL_SCAN"
        self.current_goal = None  # 'blue' or 'yellow'


        # Time tracking
        self.initial_scan_start_time = time.time()
        self.mission_start_time = time.time()
        self.blue_reached_time = None
        self.yellow_reached_time = None


        # Waypoints / paths
        self.start_position = None
        self.current_path = []    # list of (gx, gz) in grid
        self.path_idx = 0


        self.v_cmd = 0.0
        self.w_cmd = 0.0


        # Replanning bookkeeping
        self.last_plan_time = 0.0
        self.last_green_count = 0
        self.last_red_count = 0


        # Frontier bookkeeping
        self.failed_frontiers = 0
        # --- Anti-oscillation / narrow-gap helpers ---
        self.turn_lock_dir = 0         # -1 right, +1 left, 0 none
        self.turn_lock_until = 0.0
        self.last_progress_pose = None
        self.last_progress_time = time.time()



        # Semantic detection
        print("\n📸 Initializing semantic detector...")
        self.semantic_detector = None
        if self.camera:
            self.semantic_detector = CameraSemanticDetector(
                camera=self.camera,
                pose=self.pose,
                lidar=self.lidar,
                grid_resolution=self.grid.RESOLUTION,
                grid_center=self.grid.MAP_CENTER
            )
            print("✓ Semantic detector ready")
            print("  - Detects: BLUE, YELLOW, GREEN, RED")
            print("  - LiDAR-camera fusion enabled")
            print("  - Multi-frame confirmation active")
        else:
            print("⚠️ Semantic detection disabled")
        # Visualization (pygame): replaces cv2.imshow windows.
        # NOTE: We keep OpenCV for drawing into numpy arrays, but NOT for window display.
        self._pg_enabled = bool(_PYGAME_OK)
        self._pg_screen = None
        self._pg_clock = None
        self._pg_font = None
        self._pg_bigfont = None
        self._pg_last_cam = 0.0
        self._pg_last_map = 0.0
        self._pg_cam_surface = None
        self._pg_map_surface = None

        self._pygame_init()

        # Camera detection thread (logic; not visualization)
        if self.camera and self.semantic_detector:
            threading.Thread(target=self._camera_detection_loop, daemon=True).start()
            print("✓ Camera detection thread started")


        # Navigation (DWA)
        self.dwa = DynamicWindowApproach({
            'max_speed': 0.25,
            'min_speed': 0.0,
            'max_omega': 1.2,
            'acc': 0.15,
            'omega_acc': 2.0,
            'dt': TIME_STEP / 1000.0,
            'predict_time': 1.2,
            'goal_cost_gain': 1.2,
            'speed_cost_gain': 0.05,
            'wheel_base': 0.2,

            # NEW:
            'robot_radius': 0.18,                          # was 0.16
            'obstacle_cost_gain': 1.4,
            'v_samples': 7,
            'w_samples': 11,
        })



        self.initialized = True

        print("\n✅ System ready!")
        print("="*60 + "\n")
        
        
        
        
    def semantic_mask_from_color_detections(self, color: str, radius_m: float = 0.35) -> np.ndarray:
        """
        Build a boolean grid mask for a semantic color using the detections stored in semantic_detector.
        True = danger region (inside radius around each detection).
        """
        # default: no danger
        h, w = self.grid.grid_map.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if not self.semantic_detector:
            return mask

        dets = self.semantic_detector.get_all_detections(color)
        if not dets:
            return mask

        radius = max(1, int(round(radius_m / self.grid.RESOLUTION)))

        for wp in dets:
            gx, gz = int(wp.grid_x), int(wp.grid_z)

            # clamp bbox
            y0 = max(0, gz - radius)
            y1 = min(h, gz + radius + 1)
            x0 = max(0, gx - radius)
            x1 = min(w, gx + radius + 1)

            # draw filled circle
            cv2.circle(mask, (gx, gz), radius, 255, thickness=-1)

        return mask

        
    def check_frontier_progress(self):
        """Detect if robot is stuck (not moving)."""
        yaw, rx, rz = self.pose.get_relative_pose()
        now = time.time()
        if self.last_progress_pose is None:
            self.last_progress_pose = (rx, rz)
            self.last_progress_time = now
            return

        dist = math.hypot(rx - self.last_progress_pose[0], rz - self.last_progress_pose[1])
        if dist > 0.08:  # moved enough
            self.last_progress_pose = (rx, rz)
            self.last_progress_time = now

    def detect_robot_stuck(self, seconds=3.0):
        """Check if the robot is stuck for a given duration."""
        return (time.time() - self.last_progress_time) > seconds
    
    
    
    
    def detect_spin_in_place(self):
    
        """
        Detect spin-in-place: large yaw change but very little translation.
        Returns True if spinning for > ~1.2s.
        """
        yaw, rx, rz = self.pose.get_relative_pose()
        now = time.time()

        if self.spin_start_time is None:
            self.spin_start_time = now
            self.spin_yaw_ref = yaw
            self.spin_trans_ref = (rx, rz)
            return False

        dt = now - self.spin_start_time
        dyaw = abs(self._angle_diff(yaw, self.spin_yaw_ref))
        dpos = math.hypot(rx - self.spin_trans_ref[0], rz - self.spin_trans_ref[1])

        # reset if moving normally
        if dpos > 0.12:
            self.spin_start_time = now
            self.spin_yaw_ref = yaw
            self.spin_trans_ref = (rx, rz)
            return False

        # spinning if rotated a lot but didn’t move
        if dt > 1.2 and dyaw > math.radians(120) and dpos < 0.10:
            return True

        return False


    def initiate_escape_behavior(self):
        """
        Escape behavior:
        0) reverse a bit
        1) rotate hard to a chosen direction
        2) drive forward for a moment
        """
        ranges, fov, n = self._get_lidar_ranges()
        dir_choice = +1
        if ranges is not None and n > 0:
            c = n // 2
            left = np.nanmin(ranges[max(0, c-60):max(1, c-10)])
            right = np.nanmin(ranges[min(n-1, c+10):min(n, c+60)])
            dir_choice = +1 if left > right else -1

        self.escape_dir = dir_choice
        self.escape_phase = 0
        self.escape_until = time.time() + 2.0  # total budget, phases inside
        # reset spin detector refs
        self.spin_start_time = None

    def execute_escape_step(self) -> bool:
        """
        Execute one step of the existing ESCAPE sequence.
        Returns True if escape is active and we applied control this tick.
        """
        now = time.time()
        if now >= self.escape_until:
            self.escape_until = 0.0
            return False

        # Reset DWA memory while escaping (prevents stale commands after escape)
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        if self.escape_phase == 0:
            self.set_wheel_speed(-BASE_SPEED * 0.5, -BASE_SPEED * 0.5)
            if (self.escape_until - now) < 1.5:
                self.escape_phase = 1
            return True

        if self.escape_phase == 1:
            w = self.escape_dir * BASE_SPEED * 0.7
            self.set_wheel_speed(-w, +w)
            if (self.escape_until - now) < 0.9:
                self.escape_phase = 2
            return True

        if self.escape_phase == 2:
            self.set_wheel_speed(BASE_SPEED * 0.7, BASE_SPEED * 0.7)
            return True

        return False
            
        
    def lidar_scan_to_obstacle_coordinates(self, max_range=1.5, stride=2):
        """
        Convert LiDAR scan into obstacle points in WORLD coordinates (x,y),
        used by obstacle-aware DWA.
        """
        ranges, fov, n = self._get_lidar_ranges()
        if ranges is None:
            return np.empty((0, 2), dtype=np.float32)

        yaw, rx, rz = self.pose.get_relative_pose()

        # clip + subsample
        r = ranges.copy()
        r[~np.isfinite(r)] = 0.0
        r[(r <= 0.0) | (r > max_range)] = 0.0

        idx = np.arange(0, n, stride, dtype=np.int32)
        r = r[idx]
        idx = idx[r > 0.0]
        r = r[r > 0.0]
        if r.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        # angles: same convention as your mapping call (min_angle=-fov/2, step=-ang_res)
        ang_res = fov / n
        beam_ang = (-fov / 2.0) + idx.astype(np.float32) * (-ang_res)
        world_ang = yaw + beam_ang

        ox = rx + r * np.cos(world_ang)
        oy = rz + r * np.sin(world_ang)

        return np.column_stack([ox, oy]).astype(np.float32)



    # ========================================================================
    # CAMERA DETECTION THREAD
    # ========================================================================


    def _camera_detection_loop(self):
        """Continuously process camera for semantic detection (all colors)"""
        while not self.stop_event.is_set():
            if self.semantic_detector and self.state in [
                "INITIAL_SCAN", "EXPLORE", "AUTO_EXPLORE",
                "PLAN_TO_BLUE", "PLAN_TO_YELLOW",
                "NAV_TO_BLUE", "NAV_TO_YELLOW", "NAV_TO_FRONTIER"
            ]:
                try:
                    with self.map_lock:
                        occupancy_grid = self.grid.grid_map.copy()
                    self.semantic_detector.process_frame(occupancy_grid)
                    if self.semantic_detector:
                        blue = self.semantic_detector.get_waypoint('blue')
                        yellow = self.semantic_detector.get_waypoint('yellow')

                        yaw, rx, rz = self.pose.get_relative_pose()

                        blue_ok = False
                        if blue:
                            blue_ok = (math.hypot(blue.world_x - rx, blue.world_z - rz) <= BLUE_CONFIRM_DIST)

                        yellow_ok = False
                        if yellow:
                            yellow_ok = (math.hypot(yellow.world_x - rx, yellow.world_z - rz) <= YELLOW_CONFIRM_DIST)
                        
                        # Start mission as soon as BLUE is confirmed (no need YELLOW here)
                        if self.state == "AUTO_EXPLORE":
                            blue = self.semantic_detector.get_waypoint('blue')
                            yellow = self.semantic_detector.get_waypoint('yellow')
                            if blue and yellow:
                                print("\n🚀 Both pillars detected. Switching to planning.")
                                self.set_wheel_speed(0, 0)
                                self.mission_start_time = time.time()
                                # Let _state_auto_explore decide order next loop
                                # (so we don't duplicate order code here)
                                # do NOT return; keep detection thread alive
                                pass



                        # Print once when detected
                        if blue and not self.blue_seen:
                            self.blue_seen = True
                            print(f"✅ Confirmed BLUE at ({blue.world_x:.2f}, {blue.world_z:.2f}), grid ({blue.grid_x}, {blue.grid_z})")

                        if yellow and not self.yellow_seen:
                            self.yellow_seen = True
                            print(f"✅ Confirmed YELLOW at ({yellow.world_x:.2f}, {yellow.world_z:.2f}), grid ({yellow.grid_x}, {yellow.grid_z})")

                        
                        
                        
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    import traceback
                    traceback.print_exc()
            time.sleep(CAMERA_DETECTION_INTERVAL)
            
            
            
    def apply_world_boundary_to_grid(self):
        """
        Add a solid obstacle border around the occupancy grid
        so planner/explore never drives outside the mapped world.
        """
        with self.map_lock:
            g = self.grid.grid_map
            h, w = g.shape
            margin = 8  # cells (~8*0.02=0.16m). Increase if needed.
            g[:margin, :] = 0
            g[h-margin:, :] = 0
            g[:, :margin] = 0
            g[:, w-margin:] = 0



    # ========================================================================
    # PYGAME VISUALIZATION (REPLACES cv2.imshow)
    # ========================================================================

    def initialize_pygame_visualization(self):
        # Initialize pygame window. If pygame isn't available, print a warning.
        if not self._pg_enabled:
            print("⚠️ Pygame not available in this Python. Install with: python -m pip install pygame")
            return

        try:
            pygame.init()
            pygame.display.init()
            pygame.font.init()

            # Determine pane sizes
            if self.camera:
                cam_w, cam_h = int(self.camera.getWidth()), int(self.camera.getHeight())
            else:
                cam_w, cam_h = 640, 480

            map_h, map_w = self.grid.grid_map.shape  # (H,W)

            self._pg_margin = 10
            self._pg_cam_size = (cam_w, cam_h)
            self._pg_map_size = (map_w, map_h)

            # Window size: side-by-side panes (camera + map)
            win_w = self._pg_margin * 3 + cam_w + map_w
            win_h = self._pg_margin * 2 + max(cam_h, map_h) + 28  # header text row

            self._pg_screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("RosBot Visualization (pygame)")
            self._pg_clock = pygame.time.Clock()
            self._pg_font = pygame.font.SysFont("consolas", 16)
            self._pg_bigfont = pygame.font.SysFont("consolas", 18, bold=True)

            print("✅ Pygame visualization window started")

        except Exception as e:
            self._pg_enabled = False
            print(f"⚠️ Pygame init failed: {e}")

    @staticmethod
    def _bgr_to_pygame_surface(img_bgr: np.ndarray):
        # Convert a BGR uint8 image (H,W,3) to a pygame Surface.
        if img_bgr is None:
            return None
        if img_bgr.dtype != np.uint8:
            img_bgr = img_bgr.astype(np.uint8, copy=False)

        # Convert BGR -> RGB
        img_rgb = img_bgr[:, :, ::-1]

        # pygame.surfarray.make_surface expects array shape (W,H,3)
        surf = pygame.surfarray.make_surface(np.swapaxes(img_rgb, 0, 1))
        return surf

    def render_camera_with_overlays(self) -> np.ndarray:
        # Render camera frame with detection overlays (BGR).
        if not self.camera or not self.semantic_detector:
            return None

        image = self.camera.getImage()
        if not image:
            return None

        width = self.camera.getWidth()
        height = self.camera.getHeight()
        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        img_bgr = img_array[:, :, :3].copy()

        img_with = self.semantic_detector.draw_detections_on_image(img_bgr)

        cv2.putText(img_with, "Semantic Detection - 4 Colors", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        yaw, xw, zw = self.pose.get_relative_pose()
        cv2.putText(img_with, f"State: {self.state}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_with, f"Pos: ({xw:.2f}, {zw:.2f})", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img_with

    def render_semantic_map_visualization(self) -> np.ndarray:
        # Render semantic map visualization (BGR) using the same logic as cv2 version.
        with self.map_lock:
            base_grid = self.grid.grid_map.copy()

        vis = np.full((base_grid.shape[0], base_grid.shape[1], 3), 127, dtype=np.uint8)
        vis[base_grid == 255] = [255, 255, 255]
        vis[base_grid == 0] = [0, 0, 0]

        yaw, xw, zw = self.pose.get_relative_pose()
        robot_gx = int(round(xw / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        robot_gz = self.grid.MAP_CENTER - int(round(zw / self.grid.RESOLUTION))

        # START
        if self.start_position:
            start_gx, start_gz = self.start_position
            cv2.circle(vis, (start_gx, start_gz), 3, (0, 255, 0), -1)

        # Semantic objects
        if self.semantic_detector:
            for blue_wp in self.semantic_detector.get_all_detections('blue'):
                cv2.circle(vis, (blue_wp.grid_x, blue_wp.grid_z), 3, (255, 0, 0), -1)
            for yellow_wp in self.semantic_detector.get_all_detections('yellow'):
                cv2.circle(vis, (yellow_wp.grid_x, yellow_wp.grid_z), 3, (0, 255, 255), -1)
            for green_wp in self.semantic_detector.get_all_detections('green'):
                cv2.circle(vis, (green_wp.grid_x, green_wp.grid_z), 4, (0, 255, 0), -1)
            for red_wp in self.semantic_detector.get_all_detections('red'):
                cv2.circle(vis, (red_wp.grid_x, red_wp.grid_z), 3, (0, 0, 255), -1)

        # Driven trail (orange)
        if self.grid.path and len(self.grid.path) > 1:
            for i in range(len(self.grid.path) - 1):
                cv2.line(vis, self.grid.path[i], self.grid.path[i + 1], (0, 165, 255), 2)

        # Current planned path (thinner orange)
        if self.current_path and len(self.current_path) > 1:
            for i in range(len(self.current_path) - 1):
                cv2.line(vis, self.current_path[i], self.current_path[i + 1], (0, 140, 255), 1)
        if self.path_idx < len(self.current_path):
            cv2.circle(vis, self.current_path[self.path_idx], 5, (255, 0, 255), -1)

        # Robot
        cv2.circle(vis, (robot_gx, robot_gz), 6, (255, 255, 255), -1)
        cv2.circle(vis, (robot_gx, robot_gz), 7, (0, 0, 0), 2)
        heading_length = 15
        end_x = int(robot_gx + heading_length * math.cos(yaw))
        end_z = int(robot_gz - heading_length * math.sin(yaw))
        cv2.arrowedLine(vis, (robot_gx, robot_gz), (end_x, end_z),
                        (255, 255, 255), 2, tipLength=0.3)

        self._draw_legend(vis)
        return vis

    def _pygame_render_step(self):
        # Render one pygame frame (camera + map). Call this in the main Webots loop.
        if not self._pg_enabled or self._pg_screen is None:
            return

        # Handle window events (must pump)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.stop_event.set()
                return

        now = time.time()

        # Throttle camera/map rendering
        if (now - self._pg_last_cam) > 0.033:  # ~30 FPS
            cam = self._render_camera_bgr()
            self._pg_cam_surface = self._bgr_to_pygame_surface(cam) if cam is not None else None
            self._pg_last_cam = now

        if (now - self._pg_last_map) > 0.050:  # ~20 FPS
            mp = self._render_semantic_map_bgr()
            self._pg_map_surface = self._bgr_to_pygame_surface(mp) if mp is not None else None
            self._pg_last_map = now

        # Clear background
        self._pg_screen.fill((30, 30, 30))

        # Header
        if self._pg_bigfont:
            title = self._pg_bigfont.render("RosBot: Camera (left) | Semantic Map (right)", True, (230, 230, 230))
            self._pg_screen.blit(title, (self._pg_margin, self._pg_margin))

        y0 = self._pg_margin + 28
        x_cam = self._pg_margin
        x_map = self._pg_margin * 2 + self._pg_cam_size[0]

        # Blit camera
        if self._pg_cam_surface is not None:
            self._pg_screen.blit(self._pg_cam_surface, (x_cam, y0))
        else:
            if self._pg_font:
                msg = self._pg_font.render("Camera disabled / not ready", True, (200, 200, 200))
                self._pg_screen.blit(msg, (x_cam + 10, y0 + 10))

        # Blit map
        if self._pg_map_surface is not None:
            self._pg_screen.blit(self._pg_map_surface, (x_map, y0))

        pygame.display.flip()
        if self._pg_clock:
            self._pg_clock.tick(60)


    def _camera_view_loop(self):
        """Display camera view with detections"""
        while not self.stop_event.is_set():
            if not self.camera or not self.semantic_detector:
                break
            try:
                image = self.camera.getImage()
                if not image:
                    time.sleep(0.033)
                    continue
                width = self.camera.getWidth()
                height = self.camera.getHeight()
                img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
                img_bgr = img_array[:, :, :3].copy()
                img_with_detections = self.semantic_detector.draw_detections_on_image(img_bgr)
                cv2.putText(img_with_detections, "Semantic Detection - 4 Colors", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                yaw, xw, zw = self.pose.get_relative_pose()
                cv2.putText(img_with_detections, f"State: {self.state}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img_with_detections, f"Pos: ({xw:.2f}, {zw:.2f})", (10, height-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("RosBot Camera View", img_with_detections)
                cv2.waitKey(1)
            except Exception as e:
                print(f"⚠️ Camera view error: {e}")
                break
            time.sleep(0.033)


    # ========================================================================
    # SEMANTIC MAP VISUALIZATION
    # ========================================================================


    def _semantic_map_loop(self):
        """Real-time semantic map with all detected objects (small colored dots)"""
        while not self.stop_event.is_set():
            with self.map_lock:
                base_grid = self.grid.grid_map.copy()
            vis = np.full((base_grid.shape[0], base_grid.shape[1], 3),
                          127, dtype=np.uint8)
            vis[base_grid == 255] = [255, 255, 255]
            vis[base_grid == 0] = [0, 0, 0]


            yaw, xw, zw = self.pose.get_relative_pose()
            robot_gx = int(round(xw / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
            robot_gz = self.grid.MAP_CENTER - int(round(zw / self.grid.RESOLUTION))


            # START
            if self.start_position:
                start_gx, start_gz = self.start_position
                cv2.circle(vis, (start_gx, start_gz), 3, (0, 255, 0), -1)


            # Semantic objects
            if self.semantic_detector:
                for blue_wp in self.semantic_detector.get_all_detections('blue'):
                    cv2.circle(vis, (blue_wp.grid_x, blue_wp.grid_z),
                               3, (255, 0, 0), -1)
                for yellow_wp in self.semantic_detector.get_all_detections('yellow'):
                    cv2.circle(vis, (yellow_wp.grid_x, yellow_wp.grid_z),
                               3, (0, 255, 255), -1)
                for green_wp in self.semantic_detector.get_all_detections('green'):
                    cv2.circle(vis, (green_wp.grid_x, green_wp.grid_z),
                               4, (0, 255, 0), -1)
                for red_wp in self.semantic_detector.get_all_detections('red'):
                    cv2.circle(vis, (red_wp.grid_x, red_wp.grid_z),
                               3, (0, 0, 255), -1)


            # Driven trail from LidarMap (orange)
            if self.grid.path and len(self.grid.path) > 1:
                for i in range(len(self.grid.path) - 1):
                    cv2.line(vis, self.grid.path[i], self.grid.path[i + 1],
                             (0, 165, 255), 2)


            # Current planned path (thinner orange)
            if self.current_path and len(self.current_path) > 1:
                for i in range(len(self.current_path) - 1):
                    cv2.line(vis, self.current_path[i], self.current_path[i + 1],
                             (0, 140, 255), 1)
            if self.path_idx < len(self.current_path):
                cv2.circle(vis, self.current_path[self.path_idx],
                           5, (255, 0, 255), -1)


            # Robot
            cv2.circle(vis, (robot_gx, robot_gz), 6, (255, 255, 255), -1)
            cv2.circle(vis, (robot_gx, robot_gz), 7, (0, 0, 0), 2)
            heading_length = 15
            end_x = int(robot_gx + heading_length * math.cos(yaw))
            end_z = int(robot_gz - heading_length * math.sin(yaw))
            cv2.arrowedLine(vis, (robot_gx, robot_gz), (end_x, end_z),
                            (255, 255, 255), 2, tipLength=0.3)


            self._draw_legend(vis)


            cv2.imshow("Semantic Map (Real-time)", vis)
            cv2.waitKey(1)
            time.sleep(0.05)


    def _draw_legend(self, vis):
        """Draw legend with all 4 semantic colors"""
        legend_x, legend_y = 10, 10
        legend_w, legend_h = 160, 155
        cv2.rectangle(vis, (legend_x, legend_y),
                      (legend_x + legend_w, legend_y + legend_h), (40, 40, 40), -1)
        cv2.rectangle(vis, (legend_x, legend_y),
                      (legend_x + legend_w, legend_y + legend_h), (255, 255, 255), 1)
        y_offset = legend_y + 20
        cv2.putText(vis, "Semantic Map:", (legend_x + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        items = [
            ((0, 255, 0), "START"),
            ((255, 0, 0), "Blue Pillar"),
            ((0, 255, 255), "Yellow Pillar"),
            ((0, 255, 0), "Green Poison"),
            ((0, 0, 255), "Red Wall"),
            ((0, 0, 0), "Obstacle"),
            ((255, 255, 255), "Robot")
        ]
        for i, (color, label) in enumerate(items):
            y_offset += 18
            if i < 6:
                cv2.circle(vis, (legend_x + 15, y_offset - 5), 4, color, -1)
            else:
                cv2.circle(vis, (legend_x + 15, y_offset - 5), 6, color, -1)
                cv2.circle(vis, (legend_x + 15, y_offset - 5), 7, (0, 0, 0), 1)
            cv2.putText(vis, label, (legend_x + 30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


    # ========================================================================
    # MOTION CONTROL
    # ========================================================================


    def update_motion_control(self, left_speed, right_speed):
        """Set wheel velocities safely (never send NaN/inf to Webots)."""
        if not np.isfinite(left_speed) or not np.isfinite(right_speed):
            left_speed, right_speed = 0.0, 0.0

        # HARD POISON OVERRIDE (works in ALL states, even during NAV)
        if self.semantic_detector and getattr(self.semantic_detector, "green_alarm", False):
            d = int(getattr(self.semantic_detector, "green_alarm_dir", 1)) or 1
            left_speed, right_speed = (-0.25 * MAX_SPEED * d, +0.25 * MAX_SPEED * d)


        # clamp
        maxv = float(MAX_SPEED)
        left_speed = float(max(-maxv, min(maxv, left_speed)))
        right_speed = float(max(-maxv, min(maxv, right_speed)))

        self.motors[0].setVelocity(left_speed)
        self.motors[2].setVelocity(left_speed)
        self.motors[1].setVelocity(right_speed)
        self.motors[3].setVelocity(right_speed)



    def _maybe_update_map(self):
        """Update occupancy grid"""
        yaw, x, z = self.pose.get_relative_pose()
        rotating = (self.prev_yaw is not None and
                    abs(self._angle_diff(yaw, self.prev_yaw)) > ROT_THRESHOLD)
        self.prev_yaw = yaw


        self.grid.set_global_pose(x, z, yaw)


        if self.start_position is None:
            self.start_position = (
                int(round(x / self.grid.RESOLUTION)) + self.grid.MAP_CENTER,
                self.grid.MAP_CENTER - int(round(z / self.grid.RESOLUTION))
            )
            print(f"\n📍 START position marked at grid {self.start_position}")


        if not rotating:
            ranges = self.lidar.getRangeImage()
            samples = self.lidar.getHorizontalResolution()
            if samples > 0:
                fov = self.lidar.getFov()
                ang_res = fov / samples
                with self.map_lock:
                    self.grid.update_map(ranges, ang_res, -fov / 2)
                    
                    
                    
                    
    def check_goal_proximity(self, goal_color: str, dist_thresh: float, hold_sec: float) -> bool:
        if not self.semantic_detector:
            return False
        wp = self.semantic_detector.get_waypoint(goal_color)
        if not wp:
            return False

        _, rx, rz = self.pose.get_relative_pose()
        d = math.hypot(wp.world_x - rx, wp.world_z - rz)
        now = time.time()

        attr = "blue_close_since" if goal_color == "blue" else "yellow_close_since"
        t0 = getattr(self, attr)

        if d <= dist_thresh:
            if t0 is None:
                setattr(self, attr, now)
                return False
            return (now - t0) >= hold_sec
        else:
            setattr(self, attr, None)
            return False

                    
    def mark_green_as_obstacle(self):
        """Paint green poison area into occupancy grid as obstacles (0)."""
        if not self.semantic_detector:
            return

        with self.map_lock:
            g = self.grid.grid_map
            h, w = g.shape

            for green_wp in self.semantic_detector.get_all_detections('green'):
                gx, gz = green_wp.grid_x, green_wp.grid_z

                radius = int(0.45 / self.grid.RESOLUTION)  # 35cm safety bubble
                y_min, y_max = max(0, gz - radius), min(h, gz + radius + 1)
                x_min, x_max = max(0, gx - radius), min(w, gx + radius + 1)

                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        if math.hypot(x - gx, y - gz) <= radius:
                            g[y, x] = 0   # ✅ obstacle in occupancy grid



    def update_robot_trail(self):
        """Append current robot pose to LidarMap.path for orange trajectory."""
        yaw, rx, rz = self.pose.get_relative_pose()
        gx = int(round(rx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        gz = self.grid.MAP_CENTER - int(round(rz / self.grid.RESOLUTION))
        if not self.grid.path or (gx, gz) != self.grid.path[-1]:
            self.grid.path.append((gx, gz))
            if len(self.grid.path) > 5000:
                self.grid.path.pop(0)


    # ========================================================================
    # WAYPOINT & PATH PLANNING WITH SEMANTIC COSTS
    # ========================================================================


    def calculate_waypoint_for_goal_approach(self, pillar_world_x, pillar_world_z,
                                     offset=GOAL_APPROACH_OFFSET):
        """Calculate approach waypoint offset from pillar"""
        yaw, robot_x, robot_z = self.pose.get_relative_pose()
        dx = pillar_world_x - robot_x
        dz = pillar_world_z - robot_z
        distance = math.hypot(dx, dz)
        if distance < 0.01:
            return None
        dx /= distance
        dz /= distance
        approach_x = pillar_world_x - offset * dx
        approach_z = pillar_world_z - offset * dz
        approach_gx = int(round(approach_x / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        approach_gz = self.grid.MAP_CENTER - int(round(approach_z / self.grid.RESOLUTION))
        return (approach_gx, approach_gz)


    def snap_goal_to_free_space(self, goal_gx_gz, max_radius_cells=30, cost_map=None):
        """
        Snap to a cell that is FREE in grid_map AND (if cost_map provided) is traversable (finite).
        """
        if goal_gx_gz is None:
            return None
        gx0, gz0 = goal_gx_gz

        with self.map_lock:
            grid = self.grid.grid_map.copy()

        h, w = grid.shape

        def in_bounds(x, y):
            return 0 <= x < w and 0 <= y < h

        def ok(x, y):
            if int(grid[y, x]) != 255:
                return False
            if cost_map is not None and not np.isfinite(cost_map[y, x]):
                return False
            return True

        if in_bounds(gx0, gz0) and ok(gx0, gz0):
            return (gx0, gz0)

        for r in range(1, int(max_radius_cells) + 1):
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x = gx0 + dx
                    y = gz0 + dz
                    if not in_bounds(x, y):
                        continue
                    if ok(x, y):
                        return (x, y)

        return None


    def build_cost_map_with_semantics(self, base_grid: np.ndarray, unknown_mode: str = "penalize") -> np.ndarray:
        """Build a semantic cost map for path planning
        
        Args:
            base_grid: occupancy grid
            unknown_mode: "block" to treat unknowns as obstacles, "penalize" to just increase cost
        """
        cost_map = np.full_like(base_grid, 1.0, dtype=np.float32)
        cost_map[base_grid == 0] = np.inf  # Obstacles impassable
        # UNKNOWN handling: exploration wants penalize; mission wants block
        if unknown_mode == "block":
            cost_map[base_grid == 127] = np.inf
        else:
            cost_map[base_grid == 127] = 5.0


        if not self.semantic_detector:
            return cost_map

        # ------------------------------------------------------------------
        # Robot-footprint obstacle inflation (blocks too-narrow passages)
        # ------------------------------------------------------------------
        length_px, width_px = self.grid.robot()  # from LidarMap.robot()
        robot_rad_px = int(math.ceil(width_px / 2.0))

        # extra safety margin (tune 0.02–0.05 m)
        safety_m = 0.03
        safety_px = int(math.ceil(safety_m / self.grid.RESOLUTION))

        hard_rad = robot_rad_px + safety_px
        if hard_rad > 0:
            obs = (base_grid == 0).astype(np.uint8) * 255
            k = 2 * hard_rad + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            inflated = cv2.dilate(obs, kernel)

            # HARD BLOCK: anything inside robot clearance is not traversable
            cost_map[inflated > 0] = np.inf

        # Optional: add a softer “stay away from walls” band outside the hard block
        soft_m = 0.20  # tune 0.15–0.30
        soft_rad = int(round(soft_m / self.grid.RESOLUTION))
        if soft_rad > hard_rad:
            obs = (base_grid == 0).astype(np.uint8) * 255
            k2 = 2 * soft_rad + 1
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
            inflated2 = cv2.dilate(obs, kernel2)

            near = (inflated2 > 0) & np.isfinite(cost_map)
            cost_map[near] = np.maximum(cost_map[near], 8.0)



        for green_wp in self.semantic_detector.get_all_detections('green'):
            gx, gz = green_wp.grid_x, green_wp.grid_z

            radius = int(0.35 / self.grid.RESOLUTION)  # 35cm safety bubble
            y_min, y_max = max(0, gz - radius), min(cost_map.shape[0], gz + radius + 1)
            x_min, x_max = max(0, gx - radius), min(cost_map.shape[1], gx + radius + 1)

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if math.hypot(x - gx, y - gz) <= radius:
                        cost_map[y, x] = np.inf   # ✅ HARD BLOCK


        for red_wp in self.semantic_detector.get_all_detections('red'):
            gx, gz = red_wp.grid_x, red_wp.grid_z
            radius = int(0.45 / self.grid.RESOLUTION)
            y_min, y_max = max(0, gz - radius), min(cost_map.shape[0], gz + radius + 1)
            x_min, x_max = max(0, gx - radius), min(cost_map.shape[1], gx + radius + 1)
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dist = math.hypot(x - gx, y - gz)
                    if dist <= radius and cost_map[y, x] != np.inf and cost_map[y, x] < 100:
                        cost_map[y, x] = max(cost_map[y, x],
                                             50.0 * (1.0 - dist / radius))


        return cost_map


    def plan_path_to_color_goal(self, goal_color: str) -> bool:
        """Plan a path to the specified color goal"""
        if not self.semantic_detector:
            return False


        waypoint = self.semantic_detector.get_waypoint(goal_color)
        if not waypoint:
            print(f"❌ {goal_color.upper()} waypoint not available")
            return False


        approach_goal = self.calculate_waypoint_for_goal_approach(
            waypoint.world_x, waypoint.world_z
        )
        if not approach_goal:
            print(f"❌ Cannot calculate approach to {goal_color}")
            return False


        print(f"\n📍 Planning path to {goal_color.upper()}...")
        print(f"  Pillar at: ({waypoint.world_x:.2f}, {waypoint.world_z:.2f})")
        print(f"  Approach waypoint: grid {approach_goal}")


        yaw, x, z = self.pose.get_relative_pose()
        start_gx = int(round(x / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        start_gz = self.grid.MAP_CENTER - int(round(z / self.grid.RESOLUTION))

        with self.map_lock:
            grid = self.grid.grid_map.copy()

        cost_map = self.build_cost_map_with_semantics(grid, unknown_mode="block")

        # snap GOAL using cost_map (so it's not inside inflated obstacles)
        snapped_goal = self.snap_goal_to_nearest_free(approach_goal, max_radius_cells=35, cost_map=cost_map)
        if snapped_goal is None:
            print(f"❌ No reachable SAFE cell near approach goal {approach_goal} for {goal_color.upper()}")
            return False
        approach_goal = snapped_goal

        # (optional but recommended) snap START too if inflation blocks it
        if not np.isfinite(cost_map[start_gz, start_gx]):
            snapped_start = self.snap_goal_to_nearest_free((start_gx, start_gz), max_radius_cells=20, cost_map=cost_map)
            if snapped_start is None:
                print(f"❌ Start is inside inflated obstacle and no safe nearby start found")
                return False
            start_gx, start_gz = snapped_start

        planner = CostAwarePlanner(cost_map)
        path = planner.plan((start_gx, start_gz), approach_goal)
        if not path:
            print(f"❌ No path to {goal_color}")
            return False

        self.current_path = path
        self.path_idx = 0
        self.last_plan_time = time.time()
        print(f"✓ Path planned: {len(self.current_path)} waypoints")
        return True


    # ========================================================================
    # GLOBAL PATH MONITORING & REPLANNING HOOKS
    # ========================================================================


    def validate_current_path(self) -> bool:
        """Check if remaining path is still free in the updated occupancy grid"""
        if not self.current_path or self.path_idx >= len(self.current_path):
            return False
        with self.map_lock:
            grid = self.grid.grid_map.copy()
        end_idx = min(len(self.current_path), self.path_idx + PATH_LOOKAHEAD)
        for i in range(self.path_idx, end_idx, PATH_CHECK_STEP):
            gx, gz = self.current_path[i]
            if gx < 0 or gx >= grid.shape[1] or gz < 0 or gz >= grid.shape[0]:
                return False
            if grid[gz, gx] == 0:
                print(f"⚠️ Path cell {self.current_path[i]} became occupied")
                return False
        return True


    def detect_semantic_changes(self) -> bool:
        """Detect significant new semantic info (green/red)"""
        if not self.semantic_detector:
            return False
        green_n = len(self.semantic_detector.get_all_detections('green'))
        red_n = len(self.semantic_detector.get_all_detections('red'))
        changed = (green_n > self.last_green_count) or (red_n > self.last_red_count)
        if changed:
            print(f"ℹ️ Semantic change: Green {self.last_green_count}->{green_n}, "
                  f"Red {self.last_red_count}->{red_n}")
        self.last_green_count = green_n
        self.last_red_count = red_n
        return changed


    def replan_for_current_goal(self):
        """Trigger replanning if path invalid or semantics changed"""
        now = time.time()
        if now - self.last_plan_time < REPLAN_INTERVAL:
            return
        need_replan = (not self.validate_current_path()) or self.detect_semantic_changes()
        if not need_replan or not self.current_goal:
            return
        print("\n🔄 Replanning due to map/semantic change...")
        if not self.plan_path_to_color_goal(self.current_goal):
            print("⚠️ Replanning failed, keeping old path")


    # ========================================================================
    # NAVIGATION ALONG PATH (DWA)
    # ========================================================================

    def navigate_path_with_dwa(self):
        """Stable waypoint follower (P-controller). Returns True when path done."""
        if not self.current_path or self.path_idx >= len(self.current_path):
            return True

        # pick a lookahead waypoint (helps reduce circling)
        lookahead = 6
        idx = min(self.path_idx + lookahead, len(self.current_path) - 1)
        gx, gz = self.current_path[idx]

        # convert grid -> world
        goal_x = (gx - self.grid.MAP_CENTER) * self.grid.RESOLUTION
        goal_z = (self.grid.MAP_CENTER - gz) * self.grid.RESOLUTION

        yaw, rx, rz = self.pose.get_relative_pose()

        # if current waypoint reached, advance index
        cur_gx, cur_gz = self.current_path[self.path_idx]
        cur_x = (cur_gx - self.grid.MAP_CENTER) * self.grid.RESOLUTION
        cur_z = (self.grid.MAP_CENTER - cur_gz) * self.grid.RESOLUTION
        print(f"➡️ Following path: waypoint {self.path_idx}/{len(self.current_path)}")

        if math.hypot(cur_x - rx, cur_z - rz) < 0.10:
            self.path_idx += 1
            if self.path_idx >= len(self.current_path):
                return True
            return False

        # heading to goal
        target_ang = math.atan2(goal_z - rz, goal_x - rx)
        ang_err = self._angle_diff(target_ang, yaw)

        # --- Goal-following command (same as before) ---
        v_goal = 0.18
        w_goal = 1.6 * ang_err
        w_goal = max(-1.0, min(1.0, w_goal))

        if abs(ang_err) > 0.7:
            v_goal = 0.08

        # --- Obstacle-aware DWA tracking to the lookahead waypoint ---
        obs_xy = self._lidar_to_obstacles_xy(max_range=1.2, stride=2)

        state = (rx, rz, yaw)         # DWA uses (x, y, theta) ; here y := rz
        goal  = (goal_x, goal_z)

        cur_v = getattr(self, "v_cmd", 0.0)
        cur_w = getattr(self, "w_cmd", 0.0)

        (u, _) = self.dwa.dwa_control(state, goal, cur_v, cur_w, obs_xy=obs_xy)
        v, w = float(u[0]), float(u[1])

        # Respect your path-tracking speed reduction on large heading error
        v = min(v, v_goal)

        self.v_cmd = v
        self.w_cmd = w

        vl, vr = self.dwa.calc_wheel_speeds(v, w)
        scale = MAX_SPEED / max(1e-6, self.dwa.cfg['max_speed'])
        self.set_wheel_speed(vl * scale, vr * scale)
        return False

        
    def check_if_goal_reached(self, goal_color: str) -> bool:
        """Check if the specified goal has been reached."""
        if not self.semantic_detector:
            return False
        waypoint = self.semantic_detector.get_waypoint(goal_color)
        if not waypoint:
            return False
        yaw, rx, rz = self.pose.get_relative_pose()
        distance = math.hypot(waypoint.world_x - rx, waypoint.world_z - rz)
        return distance < WAYPOINT_REACHED_THRESHOLD


    # ========================================================================
    # LIDAR + MAP-AWARE LOCAL REACTIVE AVOIDANCE (AUTO_EXPLORE)
    # ========================================================================


    def _get_lidar_ranges(self):
        """
        Read current LiDAR scan as a numpy array.
        Returns (ranges, fov, n_beams) or (None, 0, 0) if not ready.
        """
        if self.lidar is None:
            return None, 0.0, 0
        ranges = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        n = self.lidar.getHorizontalResolution()
        if ranges.size == 0 or n == 0:
            return None, 0.0, 0
        fov = self.lidar.getFov()
        return ranges, fov, n


    def _lidar_sector_min(self, ranges, start_idx, end_idx):
        """
        Utility: min distance in a LiDAR index interval [start_idx, end_idx].
        Handles wrap-around and invalid values.
        """
        n = ranges.size
        if n == 0:
            return np.inf
        start_idx %= n
        end_idx %= n

        if start_idx <= end_idx:
            sector = ranges[start_idx:end_idx + 1]
        else:
            sector = np.concatenate((ranges[start_idx:], ranges[:end_idx + 1]))

        sector = sector[np.isfinite(sector)]
        sector = sector[sector > 0.0]
        if sector.size == 0:
            return np.inf
        return float(np.min(sector))


    def _unknown_fraction_side(self, side: str) -> float:
        """
        Estimate how much UNKNOWN area is on the given side ('left' or 'right')
        using the occupancy grid around the robot.
        Returns fraction in [0,1].
        """
        with self.map_lock:
            grid = self.grid.grid_map.copy()

        yaw, rx, rz = self.pose.get_relative_pose()
        h, w = grid.shape

        radius_m = 1.0
        step_m = 0.05
        unknown = 0
        total = 0

        if side == 'left':
            start_ang = yaw + math.radians(20)
            end_ang   = yaw + math.radians(160)
            ang_step  = math.radians(10)
            a = start_ang
            while a <= end_ang:
                d = 0.0
                while d <= radius_m:
                    d += step_m
                    wx = rx + d * math.cos(a)
                    wz = rz + d * math.sin(a)
                    gx = int(round(wx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
                    gz = self.grid.MAP_CENTER - int(round(wz / self.grid.RESOLUTION))
                    if 0 <= gx < w and 0 <= gz < h:
                        val = grid[gz, gx]
                        total += 1
                        if val not in (0, 255):
                            unknown += 1
                a += ang_step
        else:
            start_ang = yaw - math.radians(20)
            end_ang   = yaw - math.radians(160)
            ang_step  = math.radians(10)
            a = start_ang
            while a >= end_ang:
                d = 0.0
                while d <= radius_m:
                    d += step_m
                    wx = rx + d * math.cos(a)
                    wz = rz + d * math.sin(a)
                    gx = int(round(wx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
                    gz = self.grid.MAP_CENTER - int(round(wz / self.grid.RESOLUTION))
                    if 0 <= gx < w and 0 <= gz < h:
                        val = grid[gz, gx]
                        total += 1
                        if val not in (0, 255):
                            unknown += 1
                a -= ang_step

        if total == 0:
            return 0.0
        return unknown / total


    def _safe_heading_from_lidar(self):
        ranges, fov, n = self._get_lidar_ranges()
        if ranges is None:
            return 0.0, 0.7

        r = ranges.copy()
        r[~np.isfinite(r)] = 0.0
        r[(r <= 0.0)] = 0.0

        # ---------- Tunable thresholds ----------
        STOP_DIST = 0.40
        SLOW_DIST = 0.65

        # corridor detection (both sides close)
        CORRIDOR_SIDE_NEAR = 0.35     # if left and right both under this => narrow corridor
        DESIRED_WALL_DIST  = 0.28     # target distance to the wall when following

        CRUISE_V = 0.18
        TURN_W_HARD = 0.95
        TURN_W_SOFT = 0.55

        # lock turning direction to stop flip-flop
        LOCK_TIME = 0.8

        fov_deg = fov * 180.0 / math.pi
        def deg_to_idx(deg):
            return int(round((deg / fov_deg) * n))

        def sector_dist(i0, i1):
            i0 %= n; i1 %= n
            if i0 <= i1:
                s = r[i0:i1+1]
            else:
                s = np.concatenate((r[i0:], r[:i1+1]))
            s = s[s > 0.0]
            if s.size == 0:
                return float("inf")
            return float(np.percentile(s, 10))  # robust

        c = n // 2
        front_deg = 35
        side_deg  = 80

        hf = deg_to_idx(front_deg/2)
        hs = deg_to_idx(side_deg)

        d_front = sector_dist(c - hf, c + hf)
        d_left  = sector_dist(c - hs, c - hf)
        d_right = sector_dist(c + hf, c + hs)

        now = time.time()

        # ---------- ESCAPE if stuck ----------
        # If not moving for 3s -> rotate hard in one chosen direction for 1s
        if self._is_stuck(3.0):
            if self.turn_lock_dir == 0:
                self.turn_lock_dir = +1 if d_left > d_right else -1
            self.turn_lock_until = now + 1.0
            return 0.0, self.turn_lock_dir * TURN_W_HARD

        # ---------- Apply turn lock (hysteresis) ----------
        if now < self.turn_lock_until and self.turn_lock_dir != 0:
            # keep turning same way for a moment
            return 0.05, self.turn_lock_dir * TURN_W_SOFT

        # ---------- HARD STOP ----------
        if d_front < STOP_DIST:
            self.turn_lock_dir = +1 if d_left > d_right else -1
            self.turn_lock_until = now + LOCK_TIME
            return 0.0, self.turn_lock_dir * TURN_W_HARD

        # ---------- CORRIDOR MODE (wall-follow) ----------
        in_corridor = (d_left < CORRIDOR_SIDE_NEAR) and (d_right < CORRIDOR_SIDE_NEAR)
        if in_corridor:
            # choose one wall to follow consistently (pick the closer wall)
            follow_left = (d_left < d_right)  # follow the nearer wall
            if follow_left:
                # maintain DESIRED_WALL_DIST from left wall
                err = (DESIRED_WALL_DIST - d_left)
                w = -1.2 * err  # negative => turn right if too close to left
            else:
                err = (DESIRED_WALL_DIST - d_right)
                w = +1.2 * err  # positive => turn left if too close to right

            # clamp steering
            w = max(min(w, TURN_W_SOFT), -TURN_W_SOFT)

            # go forward slowly but consistently
            v = CRUISE_V * 0.9
            return v, w

        # ---------- SLOW AVOID ----------
        if d_front < SLOW_DIST:
            self.turn_lock_dir = +1 if d_left > d_right else -1
            self.turn_lock_until = now + 0.4
            return CRUISE_V * 0.25, self.turn_lock_dir * TURN_W_SOFT

        # ---------- CRUISE (light centering) ----------
        diff = (d_left - d_right)
        denom = max(d_left + d_right, 1e-3)
        bias = max(min(diff / denom, 1.0), -1.0)

        v = CRUISE_V
        w = 0.25 * bias
        return v, w


    def lidar_based_exploration(self):
        """Perform reactive exploration using LiDAR data."""
        v, w = self._safe_heading_from_lidar()
                # If not too close in front, never allow pure spin (v=0) in explore
        ranges, fov, n = self._get_lidar_ranges()
        if ranges is not None and n > 0:
            c = n // 2
            front = np.array(ranges[c-10:c+10], dtype=np.float32)
            front = front[np.isfinite(front)]
            front = front[front > 0.0]
            if front.size > 0 and np.percentile(front, 10) > 0.55:
                v = max(v, 0.08)  # force gentle forward motion

        
        vl, vr = self.dwa.calc_wheel_speeds(v, w)
        scale = MAX_SPEED / max(1e-6, self.dwa.cfg['max_speed'])
        self.set_wheel_speed(vl * scale, vr * scale)


    
    def servo_to_detected_pillar(self, color: str) -> bool:
        """
        If the detector sees a decent BLUE/YELLOW candidate (NOT yet confirmed),
        steer toward it to make it fill the camera (so confirmation becomes possible).

        Returns True if we applied control this step (i.e., caller should NOT run normal explore).
        """
        if not self.semantic_detector or not self.camera:
            return False

        cand = self.semantic_detector.get_candidate(color)
        if not cand:
            return False

        score = float(cand.get("score", 0.0))
        if score < getattr(self.semantic_detector, "pillar_candidate_score_min", 0.55):
            return False  # too weak (your Image-3 case)

        bearing = float(cand.get("pixel_bearing", 0.0))  # rad, + = right side
        depth_val = cand.get("depth", None)
        depth = float(depth_val) if depth_val is not None else 2.0
        full = bool(cand.get("full", False))
        score = float(cand.get("score", 0.0))

        # --- SMOOTHER MOTION LOGIC to prevent oscillation and jerky motion ---
        # Part 2 Fix: Reduce turning gains and freeze when near-perfect

        # 1. If it looks "Full" or nearly full (high score), FREEZE to allow confirmation
        #    This fixes the "I see it but don't mark it" issue caused by continuous motion
        if full or score > 0.8:
            v = 0.0
            # Very slow, precise alignment (reduced from 1.2 to 0.5)
            w = 0.5 * bearing

        # 2. Approach Phase: Not centered enough, turn in place
        elif abs(bearing) > 0.2:
            # If we are not centered, turn in place (don't drive forward yet)
            v = 0.0
            w = 0.8 * bearing  # Reduced from 1.0/1.8 to prevent oscillation

        # 3. Centered approach: Move forward with gentle course correction
        else:
            # We are centered, drive forward smoothly
            # Slow down as we get closer to prevent crashing/clipping
            if depth > 1.0:
                v = 0.15
            elif depth > 0.5:
                v = 0.08
            else:
                v = 0.05

            # Gentle course correction while moving (reduced from 1.8)
            w = 0.6 * bearing

        vl, vr = self.dwa.calc_wheel_speeds(v, w)
        scale = MAX_SPEED / max(1e-6, self.dwa.cfg['max_speed'])
        self.set_wheel_speed(vl * scale, vr * scale)
        return True


# ========================================================================
    # STATE MACHINE
    # ========================================================================


    def execute_initial_scan(self):
        """Perform the initial scan of the environment."""
        self.set_wheel_speed(0, 0)
        elapsed = time.time() - self.initial_scan_start_time
        if elapsed >= INITIAL_SCAN_DURATION:
            print(f"\n⏱️ Initial scan finished ({elapsed:.1f}s). Starting AUTO_EXPLORE.\n")
            self.state = "AUTO_EXPLORE"


    def explore_environment(self):
        """Explore the environment systematically."""
        key = self.keyboard.getKey()
        if key == self.KEY_W:
            self.set_wheel_speed(BASE_SPEED, BASE_SPEED)
        elif key == self.KEY_S:
            self.set_wheel_speed(-BASE_SPEED, -BASE_SPEED)
        elif key == self.KEY_A:
            self.set_wheel_speed(-BASE_SPEED / 2, BASE_SPEED / 2)
        elif key == self.KEY_D:
            self.set_wheel_speed(BASE_SPEED / 2, -BASE_SPEED / 2)
        elif key == self.KEY_SPACE:
            if (self.semantic_detector and
                self.semantic_detector.has_waypoint('blue') and
                    self.semantic_detector.has_waypoint('yellow')):
                self.set_wheel_speed(0, 0)
                print("\n🚀 Starting autonomous navigation to BLUE/YELLOW!")
                self.state = "PLAN_TO_BLUE"
                self.mission_start_time = time.time()
            else:
                print("\n⚠️ Not ready - need BLUE and YELLOW pillars detected")
        elif key == self.KEY_E:
            self.set_wheel_speed(0, 0)
            print("\n🧭 Switching to AUTO_EXPLORE (LiDAR + map-aware)")
            self.state = "AUTO_EXPLORE"
        else:
            self.set_wheel_speed(0, 0)


    # ========================================================================
    # FRONTIER EXPLORATION (SYSTEMATIC COVERAGE)
    # ========================================================================

    def get_frontier_centroids(self):
        """Get the centroids of frontiers for exploration."""
        mask = self.frontier_extractor.generate_frontier_mask()
        n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        pts = []
        for i in range(1, n_lbl):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < int(self.frontier_extractor.min_frontier_cells):
                continue
            cx, cy = centroids[i]
            gx, gz = int(round(cx)), int(round(cy))
            pts.append((gx, gz, area))
        return pts

    def plan_path_to_grid_goal(self, goal_gx_gz, *, unknown_mode="penalize"):
        """Plan a path to a grid goal."""
        if goal_gx_gz is None:
            return False
        yaw, rx, rz = self.pose.get_relative_pose()
        start_gx = int(round(rx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        start_gz = self.grid.MAP_CENTER - int(round(rz / self.grid.RESOLUTION))
        start = (start_gx, start_gz)

        with self.map_lock:
            grid = self.grid.grid_map.copy()
        cost_map = self.build_cost_map_with_semantics(grid, unknown_mode=unknown_mode)
        planner = CostAwarePlanner(cost_map)
        path = planner.plan(start, goal_gx_gz)
        if not path:
            return False

        self.current_path = path
        self.path_idx = 0
        self.last_plan_time = time.time()
        return True

    def pick_new_frontier_goal(self):
        """Pick a new frontier goal for exploration."""
        now = time.time()
        if (now - self.frontier_last_pick) < 1.0:
            return False

        candidates = self.get_frontier_centroids()
        if not candidates:
            return False

        yaw, rx, rz = self.pose.get_relative_pose()
        rgx = int(round(rx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        rgz = self.grid.MAP_CENTER - int(round(rz / self.grid.RESOLUTION))

        scored = []
        for gx, gz, area in candidates:
            if (gx, gz) in self.frontier_blacklist:
                continue
            d = math.hypot(gx - rgx, gz - rgz)
            if d < 15:
                continue
            # prioritize larger frontiers and coverage
            score = 0.7 * float(area) + d
            scored.append((score, (gx, gz)))

        if not scored:
            return False

        scored.sort(reverse=True, key=lambda x: x[0])

        # try top K
        for _, goal in scored[:10]:
            goal = self.snap_goal_to_nearest_free(goal, max_radius_cells=30)
            if goal is None:
                continue
            if self.plan_path_to_grid_goal(goal, unknown_mode="penalize"):
                self.frontier_goal = goal
                self.frontier_last_pick = now
                self.state = "NAV_TO_FRONTIER"
                print(f"🧭 New frontier goal: {goal}")
                return True

        # if nothing works, blacklist the best candidate
        self.frontier_blacklist.add(scored[0][1])
        self.frontier_last_pick = now
        return False


    def auto_explore_state(self):
        """Handle the AUTO_EXPLORE state."""
        # If spinning, trigger escape
        if self._detect_spin():
            print("🌀 Detected spin loop -> ESCAPE")
            self._start_escape()

        # Run escape phases if active
        now = time.time()
        if self._run_escape_step():
            return

        # if now < self.escape_until:
        #     # phase durations
        #     if self.escape_phase == 0:
        #         # reverse for 0.5s
        #         self.set_wheel_speed(-BASE_SPEED * 0.5, -BASE_SPEED * 0.5)
        #         if (self.escape_until - now) < 1.5:
        #             self.escape_phase = 1
        #         return

            if self.escape_phase == 1:
                # rotate hard for 0.6s
                w = self.escape_dir * BASE_SPEED * 0.7
                self.set_wheel_speed(-w, +w)
                if (self.escape_until - now) < 0.9:
                    self.escape_phase = 2
                return

            if self.escape_phase == 2:
                # forward for 0.9s
                self.set_wheel_speed(BASE_SPEED * 0.7, BASE_SPEED * 0.7)
                return

        # escape finished
        self.escape_until = 0.0

        # ✅ Close-confirm mission trigger (keep your close-confirm logic if you added it)
        if self.semantic_detector:
            blue = self.semantic_detector.get_waypoint('blue')
            yellow = self.semantic_detector.get_waypoint('yellow')
            yaw, rx, rz = self.pose.get_relative_pose()
            blue_ok = blue and (math.hypot(blue.world_x - rx, blue.world_z - rz) <= 0.65)
            yellow_ok = yellow and (math.hypot(yellow.world_x - rx, yellow.world_z - rz) <= 0.65)
            

        # --- Pillar discovery + confirmation (NO pillar GT cheating) ---
        # Requirement:
        #   1) Explore the environment and CONFIRM BOTH pillars first.
        #   2) Only after BOTH are confirmed, start mission planning BLUE -> YELLOW.
        #   3) While exploring, if a strong candidate is visible, servo toward it to get FULL confirmation.
        if self.semantic_detector:
            blue_known = self.semantic_detector.has_waypoint('blue')
            yellow_known = self.semantic_detector.has_waypoint('yellow')

            if hasattr(self, "plan_retry_after") and time.time() < self.plan_retry_after:
                self._lidar_reactive_explore()
                return

            # If both confirmed, start mission planning (BLUE -> YELLOW)
            if blue_known and yellow_known:
                print("\n🚀 Both pillars confirmed. Starting mission planning BLUE → YELLOW.")
                self.set_wheel_speed(0, 0)
                self.mission_start_time = time.time()
                self.state = "PLAN_TO_BLUE"
                return

            # If only one is known, focus on discovering the other (do NOT plan yet)
            if not blue_known:
                if self._servo_to_pillar_candidate('blue'):
                    return
            if not yellow_known:
                if self._servo_to_pillar_candidate('yellow'):
                    return

        # fallback: normal roaming / exploration
        self._lidar_reactive_explore()


    def navigate_to_frontier_state(self):
        """Handle the NAV_TO_FRONTIER state."""
        # If no goal/path, return to exploration
        if self.frontier_goal is None or not self.current_path:
            self.state = "AUTO_EXPLORE"
            return

        # Stuck handling
        if self._is_stuck(4.0) or self._detect_spin():
            print("⚠️ Stuck on frontier navigation -> blacklist and pick new")
            self.frontier_blacklist.add(tuple(self.frontier_goal))
            self.frontier_goal = None
            self.current_path = []
            self.path_idx = 0
            self.plan_retry_after = time.time() + 3.0
            self.state = "AUTO_EXPLORE"
            return

        done = self._navigate_along_path()
        if done:
            # reached end; mark visited and pick another
            print(f"✓ Reached frontier {self.frontier_goal}")
            self.frontier_blacklist.add(tuple(self.frontier_goal))
            self.frontier_goal = None
            self.current_path = []
            self.path_idx = 0
            self.state = "AUTO_EXPLORE"
            return

        # reached vicinity check (grid distance)
        yaw, rx, rz = self.pose.get_relative_pose()
        rgx = int(round(rx / self.grid.RESOLUTION)) + self.grid.MAP_CENTER
        rgz = self.grid.MAP_CENTER - int(round(rz / self.grid.RESOLUTION))
        if math.hypot(rgx - self.frontier_goal[0], rgz - self.frontier_goal[1]) <= self.frontier_reach_radius:
            print(f"✓ Arrived near frontier {self.frontier_goal}")
            self.frontier_blacklist.add(tuple(self.frontier_goal))
            self.frontier_goal = None
            self.current_path = []
            self.path_idx = 0
            self.state = "AUTO_EXPLORE"
            return


    def plan_to_blue_state(self):
        """Handle the PLAN_TO_BLUE state."""
        if self._plan_path_to_goal('blue'):
            self.current_goal = 'blue'
            self.state = "NAV_TO_BLUE"
        else:
            print("⚠️ Planning to BLUE failed; returning to AUTO_EXPLORE to map more")
            self.plan_retry_after = time.time() + 2.0
            self.state = "AUTO_EXPLORE"


    def navigate_to_blue_state(self):
        """Handle the NAV_TO_BLUE state."""
        self._maybe_replan_current_goal()
        # Trigger escape in NAV too (same logic as AUTO_EXPLORE)
        if self._detect_spin() or self._is_stuck(3.0):
            if time.time() >= self.escape_until:
                print("🌀 NAV stuck/spin -> ESCAPE")
                self._start_escape()

        if self._run_escape_step():
            return

        # If we are not making progress, force replanning or go back to exploration
        if self._is_stuck(4.0):
            print("⚠️ Stuck during navigation -> forcing replan / explore")
            self.last_progress_time = time.time()
            self.last_plan_time = 0.0
            self._maybe_replan_current_goal()
            if not self._path_is_valid():
                print("🧭 No valid path available; returning to AUTO_EXPLORE to explore/map.")
                self.current_path = []
                self.path_idx = 0
                self.state = "AUTO_EXPLORE"
                return
        path_complete = self._navigate_along_path()

        # If BLUE reached (either path done or distance check)
        blue_reached = self._goal_close_for("blue", dist_thresh=0.25, hold_sec=0.6)
        if path_complete or blue_reached:


            # First time reaching BLUE: stop + start 2s timer
            if self.blue_pause_until == 0.0:
                self.set_wheel_speed(0, 0)
                self.blue_reached_time = time.time()
                elapsed = self.blue_reached_time - self.mission_start_time
                print(f"\n{'='*60}")
                print("✅ BLUE PILLAR REACHED!")
                print("⏸️ Waiting 2 seconds at BLUE...")
                print(f"  Time elapsed: {elapsed:.1f}s")
                print(f"{'='*60}\n")
                self.blue_pause_until = time.time() + 2.0
                return

            # After 2 seconds -> go plan to yellow
            if time.time() >= self.blue_pause_until:
                self.blue_pause_until = 0.0
                self.state = "PLAN_TO_YELLOW"
                return

            # Still waiting
            self.set_wheel_speed(0, 0)
            return



    def plan_to_yellow_state(self):
        """Handle the PLAN_TO_YELLOW state."""
        if self._plan_path_to_goal('yellow'):
            self.current_goal = 'yellow'
            self.state = "NAV_TO_YELLOW"
        else:
            print("⚠️ Planning to YELLOW failed; returning to AUTO_EXPLORE to map more")
            self.state = "AUTO_EXPLORE"


    def navigate_to_yellow_state(self):
        """Handle the NAV_TO_YELLOW state."""
        self._maybe_replan_current_goal()
        # Trigger escape in NAV too (same logic as AUTO_EXPLORE)
        if self._detect_spin() or self._is_stuck(3.0):
            if time.time() >= self.escape_until:
                print("🌀 NAV stuck/spin -> ESCAPE")
                self._start_escape()

        if self._run_escape_step():
            return

        # If we are not making progress, force replanning or go back to exploration
        if self._is_stuck(4.0):
            print("⚠️ Stuck during navigation -> forcing replan / explore")
            self.last_progress_time = time.time()
            self.last_plan_time = 0.0
            self._maybe_replan_current_goal()
            if not self._path_is_valid():
                print("🧭 No valid path available; returning to AUTO_EXPLORE to explore/map.")
                self.current_path = []
                self.path_idx = 0
                self.state = "AUTO_EXPLORE"
                return
        path_complete = self._navigate_along_path()
        
        yellow_reached = self._goal_close_for("yellow", dist_thresh=0.28, hold_sec=0.8)
        if path_complete or yellow_reached:

            self.set_wheel_speed(0, 0)
            self.v_cmd = 0.0                    # ✅ reset DWA state
            self.w_cmd = 0.0
            self.yellow_reached_time = time.time()
            total_time = self.yellow_reached_time - self.mission_start_time
            leg2_time = self.yellow_reached_time - self.blue_reached_time
            print(f"\n{'='*60}")
            print("🎉 MISSION COMPLETE!")
            print(f"  BLUE reached: {self.blue_reached_time - self.mission_start_time:.1f}s")
            print(f"  YELLOW reached: {total_time:.1f}s")
            print(f"  Second leg: {leg2_time:.1f}s")
            print(f"{'='*60}\n")
            self.state = "DONE"


    # ========================================================================
    # MAIN LOOP
    # ========================================================================


    def run(self):
        """Main control loop"""
        print("="*60)
        print("🎮 CONTROLS (for debug):")
        print("  W/A/S/D - Manual driving (EXPLORE mode)")
        print("  SPACE   - Start BLUE→YELLOW mission (when pillars known)")
        print("  E       - Start AUTO_EXPLORE (LiDAR + map-aware)")
        print("="*60)
        print(f"\n📍 Phase 0: INITIAL_SCAN for {INITIAL_SCAN_DURATION:.1f}s (robot does not move)\n")
        print("📍 Then Phase 1: AUTO_EXPLORE (LiDAR + map-aware) starts automatically\n")


        while self.supervisor.step(TIME_STEP) != -1:
            self._maybe_update_map()
            self._update_trail()
            self.check_frontier_progress()
            #self._apply_world_fence_to_grid()
            #self._stamp_green_as_obstacle()



            # Pygame visualization update (replaces cv2.imshow)
            self._pygame_render_step()


            if self.state == "INITIAL_SCAN":
                self.execute_initial_scan()
            elif self.state == "EXPLORE":
                self.explore_environment()
            elif self.state == "AUTO_EXPLORE":
                self.auto_explore_state()
            elif self.state == "NAV_TO_FRONTIER":
                self.navigate_to_frontier_state()
            elif self.state == "PLAN_TO_BLUE":
                self.plan_to_blue_state()
            elif self.state == "NAV_TO_BLUE":
                self.navigate_to_blue_state()
            elif self.state == "PLAN_TO_YELLOW":
                self.plan_to_yellow_state()
            elif self.state == "NAV_TO_YELLOW":
                self.navigate_to_yellow_state()
            elif self.state == "DONE":
                self.set_wheel_speed(0, 0)
                self.v_cmd = 0.0
                self.w_cmd = 0.0
                self.stop_event.set()
                break


    @staticmethod
    def _angle_diff(a, b):
        """Compute angle difference"""
        return (a - b + math.pi) % (2 * math.pi) - math.pi



# ============================================================================
# ENTRY POINT
# ============================================================================



if __name__ == "__main__":
    try:
        
        controller = RosbotAutonomousNavigator()
        if getattr(controller, "initialized", False):
            controller.run()
        else:
            print("⚠️ Controller not initialized (missing devices). Exiting cleanly.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals() and hasattr(controller, 'stop_event'):
            controller.stop_event.set()
        if _PYGAME_OK:
            try:
                pygame.quit()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("\n👋 Mission ended\n")