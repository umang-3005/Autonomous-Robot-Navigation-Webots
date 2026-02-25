# RosBot Fully Automated Semantic Navigation (Webots)
**Mission:** `START → (discover & confirm) BLUE pillar → YELLOW pillar`  
**Perception:** LiDAR + RGB camera semantic detection (BLUE / YELLOW / GREEN / RED)  
**Planning:** Cost-aware A* (global) + obstacle-aware DWA (local)  
**Exploration:** Reactive LiDAR roaming with “servo-to-candidate” to confirm pillars  
**Visualization:** Pygame window (camera + semantic map), OpenCV drawing (no mandatory OpenCV windows)

This repository contains a complete Webots controller that autonomously explores, detects both pillars, and then executes the mission BLUE→YELLOW while avoiding “green poison” (hard blocked) and penalizing “red walls”. 

---

## 1) Folder structure

Place these files inside your Webots controller folder, e.g.:


**Entry point:** `my_controller.py` (creates `RosbotAutonomousNavigator` and runs it). 

---

## 2) Requirements

### Webots
- Webots R2025a+ (any version that supports Python controllers and Supervisor API should work).

### Python packages
Your Webots Python environment must have:
- `numpy`
- `opencv-python` (imported as `cv2`)
- `pygame` *(optional, recommended for visualization; code runs without it)*

`my_controller.py` will automatically disable pygame visualization if it is not available. 

---

## 3) Webots world / robot requirements

Your Webots world must contain a robot with **DEF name**:
- `rosbot` 

### Devices expected on the robot
**Motors** (exact names):
- `fl_wheel_joint`, `fr_wheel_joint`, `rl_wheel_joint`, `rr_wheel_joint` 

**LiDAR** (the controller tries multiple names and uses the first it finds):
- tries: `laser`, `lidar`, `LDS-01`, `LDS_01`, `lds-01`, `hokuyo`, `Hokuyo` 

**Camera** (optional but required for semantic mission):
- device name: `camera rgb` 

If the camera is not found, the system runs mapping/navigation without semantic detection (mission cannot complete). 

---

## 4) How to run (ready-to-submit quick guide)

### Step A — Copy files
Copy all provided `.py` files into:


### Step B — Set controller in Webots
In Webots:
1. Click the robot node (DEF `rosbot`)
2. In the robot’s `controller` field, set it to:
   - `my_controller` (folder name)

### Step C — Run simulation
Press **Run** ▶ in Webots.

On start, you should see logs like:
- “RosBot Fully Automated Semantic Navigation”
- LiDAR initialized
- Camera initialized (if present)
- “System ready” 

---

## 5) Controls (for debug)



**Default behavior:** The robot starts with `INITIAL_SCAN` (stands still for ~2 seconds), then automatically switches to `AUTO_EXPLORE`. 0}

---

## 6) System overview (what each file does)

### `my_controller.py`
Main controller + state machine:
- `INITIAL_SCAN` → `AUTO_EXPLORE` (discover both pillars) → `PLAN_TO_BLUE` → `NAV_TO_BLUE` → `PLAN_TO_YELLOW` → `NAV_TO_YELLOW` → `DONE`
- Anti-spin detection + escape routine
- Global planning: builds semantic cost map and runs A*
- Local planning: obstacle-aware DWA using LiDAR point obstacles
- Visualization: pygame camera+map display (if available) 1}

### `lidargrid.py`
Occupancy-grid mapping using **log-odds** with:
- Unknown cells = 127, Free = 255, Occupied = 0
- Ray-carving free space + hit updates
- Two-hit rule thresholding for occupancy
- Grid resolution & size set by controller (currently `resolution=0.02`, `grid_size=500`) 

### `camera_semantic_detector.py`
Semantic perception (camera + LiDAR fusion) for:
- `blue`, `yellow` pillars
- `green` poison (close-range alarm behavior supported in controller)
- `red` walls
- Produces world coordinates + grid coordinates waypoints used by planner and map visualization 3}

### `cost_aware_planner.py`
Global planner (A* on a cost map):
- 8-connected movement + octile heuristic
- Prevents diagonal corner-cutting
- Supports cost penalties and hard-blocked regions
- Path simplification via Ramer–Douglas–Peucker (optional) 4}

### `dwa.py`
Local planner (Dynamic Window Approach):
- Simulates trajectories and scores them
- **Obstacle-aware**: rejects trajectories colliding within `robot_radius`, otherwise prefers more clearance
- Returns best `(v, omega)` and trajectory 5}

### `frontier.py`
Frontier extraction helper:
- Detects frontier cells: free space adjacent to unknown
- Provides centroid targets and optional visualization overlay logic 6}

### `POSE.py`
Pose estimation from Webots robot node:
- Computes yaw + relative translation from start pose (used for mapping & navigation) 7}

---

## 7) Mission logic (important behavior)

### Phase 0 — Initial scan (robot does not move)
- `INITIAL_SCAN_DURATION = 2.0s` 8}

### Phase 1 — AUTO_EXPLORE (discover & confirm pillars)
- Robot roams using LiDAR reactive navigation.
- If the camera sees a strong candidate pillar, the robot **servos toward it** to get a full/confirmed detection.
- The mission **does not start** until **both** BLUE and YELLOW are confirmed. 9}

### Phase 2 — Planning and navigation (BLUE then YELLOW)
- Builds a semantic cost map:
  - obstacles: blocked
  - unknown: blocked for mission planning (`unknown_mode="block"`)
  - green poison: **hard blocked**
  - red walls: **high cost penalty**
  - obstacle inflation based on robot footprint + safety margin 
- Plans path using A*
- Tracks path using obstacle-aware DWA (LiDAR point obstacles) 
- At BLUE: stops ~2 seconds, then plans to YELLOW. 2}

---

## 8) Key configuration (tuning)

All major parameters are at the top of `my_controller.py`: 3}

### Simulation & motion
- `TIME_STEP = 32`
- `MAX_SPEED = 10`
- `BASE_SPEED = 4`

### Detection confirmation distances
- `BLUE_CONFIRM_DIST = 0.65`
- `YELLOW_CONFIRM_DIST = 0.65`

### Global planning / replanning
- `REPLAN_INTERVAL = 2.0`
- `PATH_LOOKAHEAD = 40`
- `PATH_CHECK_STEP = 1`

### Goal handling
- `WAYPOINT_REACHED_THRESHOLD = 0.15`
- `GOAL_APPROACH_OFFSET = 0.25`

### DWA parameters (local planner)
Inside `DynamicWindowApproach({...})`:
- `robot_radius` (collision radius)
- `obstacle_cost_gain`
- `v_samples`, `w_samples`
- `predict_time`, `max_speed`, `max_omega` 

---

## 9) Visualization

### Pygame (recommended)
If `pygame` is installed, a single window shows:
- Left: camera frame + detection overlays
- Right: semantic occupancy map + robot + paths + detections 5}


## 10) Troubleshooting

### Robot does not move at start
This is expected for the first ~2 seconds (`INITIAL_SCAN`). 7}

### “FATAL: Robot DEF 'rosbot' not found”
Your robot DEF name must be exactly `rosbot`. Update either:
- your world (rename DEF), **or**
- change `robot_def_name = "rosbot"` in `my_controller.py`. 8}

### “FATAL: LiDAR device not found”
Ensure your LiDAR device name matches one of:
`laser, lidar, LDS-01, LDS_01, lds-01, hokuyo, Hokuyo`. 9}

### Camera not found / mission never completes
Your camera device must be named `camera rgb`. 0}

### Pygame not available
Install in your Webots Python:
```bash
python -m pip install pygame
