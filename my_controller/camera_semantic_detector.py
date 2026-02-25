# camera_semantic_detector.py
"""
Production-Grade Semantic Detection System for Autonomous Navigation
- LiDAR-camera fusion for accurate 3D localization
- Multi-frame confirmation to eliminate noise
- Detects: BLUE pillars, YELLOW pillars, GREEN poison, RED walls
- START/BLUE/YELLOW/GREEN/RED waypoint management

IMPORTANT UPDATE (pillar gating):
- BLUE/YELLOW are NOT written to the semantic map until the pillar is "fully visible"
  (large bbox, near-bottom, not clipped at edges, reasonably centered).
- Partial / far glimpses (like your Image-3) will NOT create a waypoint.
- The detector still exposes a "candidate" (bearing + depth) so the controller can
  approach and center the pillar to get a full confirmation.
"""

import numpy as np
import cv2
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any


@dataclass
class ColorBlobDetection:
    """Single color blob detection"""
    centroid_u: float
    centroid_v: float
    color: str
    color_id: int
    area: float
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


@dataclass
class ConfirmedWaypoint:
    """Confirmed semantic waypoint"""
    world_x: float
    world_z: float
    grid_x: int
    grid_z: int
    color: str
    confidence: float
    confirmed_at: float


class MultiFrameConsensus:
    """
    Multi-frame consensus filter - requires N consistent observations.
    """
    def __init__(self, min_observations=5, max_variance=0.10, timeout=3.0):
        self.detections = defaultdict(list)  # {color: [(x, z, timestamp)]}
        self.min_observations = int(min_observations)
        self.max_variance = float(max_variance)  # meters^2-ish (we use var sum)
        self.timeout = float(timeout)  # seconds

    def add_observation(self, color: str, world_x: float, world_z: float) -> Tuple[bool, float, float, float]:
        current_time = time.time()

        self.detections[color].append((world_x, world_z, current_time))
        self.detections[color] = [(x, z, t) for x, z, t in self.detections[color]
                                  if current_time - t < self.timeout]

        observations = self.detections[color]
        if len(observations) < self.min_observations:
            return False, 0.0, 0.0, 0.0

        positions = np.array([(x, z) for x, z, _ in observations], dtype=np.float32)
        mean_pos = positions.mean(axis=0)
        var = np.var(positions, axis=0).sum()  # sum of variances in x and z

        if var > self.max_variance:
            return False, 0.0, 0.0, 0.0

        confidence = min(1.0, len(observations) / (self.min_observations + 5.0)) * (1.0 - min(1.0, var / self.max_variance))
        return True, float(mean_pos[0]), float(mean_pos[1]), float(confidence)


class SemanticDetector:
    """
    Semantic detector with LiDAR fusion.
    Detects: blue, yellow, green, red.

    New behavior for pillars:
      - We keep a "candidate" (bearing+depth) for approach/centering.
      - We ONLY confirm/map BLUE/YELLOW when "full pillar visible" gate passes.
    """

    COLOR_BLUE = 'blue'
    COLOR_YELLOW = 'yellow'
    COLOR_GREEN = 'green'
    COLOR_RED = 'red'

    def __init__(self, camera, pose, lidar, grid_resolution=0.02, grid_center=250):
        self.camera = camera
        self.pose = pose
        self.lidar = lidar
        self.resolution = float(grid_resolution)
        self.center = int(grid_center)

        # Near-only gating is useful for poison/walls, but too aggressive for pillars.
        self.near_only_colors = {
            'green': 0.60,
            'red':   0.50
        }

        self.color_ranges = {
            'blue':   {'lower': np.array([100, 100, 50]), 'upper': np.array([130, 255, 255])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
            'green':  {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
            'red': {
                'lower1': np.array([0, 100, 100]),   'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]), 'upper2': np.array([180, 255, 255])
            }
        }

        self.min_blob_area = 200
        self.edge_margin = 40
        self.min_lidar_range = 0.05
        self.max_lidar_range = 2.5
        self.max_angle_diff = 0.20

        self.cam_offset_forward = 0.08

        self.pillar_full_h_ratio = 0.30
        self.pillar_full_w_ratio = 0.08
        self.pillar_bottom_ratio = 0.50
        self.pillar_center_tol = 0.35
        self.pillar_candidate_score_min = 0.55

        self.confirmations = {
            'blue':   MultiFrameConsensus(min_observations=6, max_variance=0.15, timeout=2.5),
            'yellow': MultiFrameConsensus(min_observations=5, max_variance=0.15, timeout=4.0),
            'green':  MultiFrameConsensus(min_observations=4, max_variance=0.10, timeout=1.2),
            'red':    MultiFrameConsensus(min_observations=4, max_variance=0.10, timeout=1.2),
        }

        self.semantic_detections: Dict[str, List[ConfirmedWaypoint]] = {
            'blue': [], 'yellow': [], 'green': [], 'red': []
        }

        self._latest_candidate: Dict[str, Optional[Dict[str, Any]]] = {
            'blue': None, 'yellow': None, 'green': None, 'red': None
        }

        self.frame_count = 0
        self.detection_count = {'blue': 0, 'yellow': 0, 'green': 0, 'red': 0}

        self.green_alarm = False
        self.green_alarm_dir = 0
        self.green_alarm_t = 0.0

    def get_candidate(self, color: str, max_age: float = 0.6) -> Optional[Dict[str, Any]]:
        candidate = self._latest_candidate.get(color)
        if not candidate:
            return None
        if (time.time() - float(candidate.get("t", 0.0))) > max_age:
            return None
        return candidate

    def get_waypoint(self, color: str) -> Optional[ConfirmedWaypoint]:
        detections = self.semantic_detections.get(color, [])
        return detections[0] if detections else None

    def has_waypoint(self, color: str) -> bool:
        return len(self.semantic_detections.get(color, [])) > 0

    def get_all_detections(self, color: str) -> List[ConfirmedWaypoint]:
        return self.semantic_detections.get(color, [])

    def process_frame(self, occupancy_grid: np.ndarray) -> bool:
        if not self.camera:
            return False

        self.frame_count += 1
        new_confirmation = False

        for k in self._latest_candidate.keys():
            self._latest_candidate[k] = None

        image = self.camera.getImage()
        if not image:
            return False

        width = self.camera.getWidth()
        height = self.camera.getHeight()

        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        img_bgr = img_array[:, :, :3].copy()
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        self.green_alarm = False
        self.green_alarm_dir = 0

        green_mask = cv2.inRange(img_hsv,
                                 self.color_ranges['green']['lower'],
                                 self.color_ranges['green']['upper'])
        h, w = green_mask.shape
        y0 = int(0.55 * h)
        x0 = int(0.15 * w)
        x1 = int(0.85 * w)
        roi = green_mask[y0:h, x0:x1]
        ratio = float(np.count_nonzero(roi)) / float(roi.size + 1e-9)

        if ratio > 0.003:
            self.green_alarm = True
            self.green_alarm_t = time.time()
            if np.count_nonzero(roi) > 0:
                mean_x = float(np.mean(np.where(roi > 0)[1]))
                self.green_alarm_dir = (-1 if mean_x < (roi.shape[1] * 0.5) else +1)
            else:
                self.green_alarm_dir = 0

        yaw, robot_x, robot_z = self.pose.get_relative_pose()

        lidar_ranges = self.lidar.getRangeImage()
        lidar_fov = self.lidar.getFov()
        lidar_samples = self.lidar.getHorizontalResolution()
        if lidar_samples <= 0:
            return False

        lidar_angles = np.linspace(yaw - lidar_fov / 2, yaw + lidar_fov / 2, lidar_samples)

        for color_name in ['blue', 'yellow', 'green', 'red']:
            detections = self._detect_color_blobs(img_hsv, color_name, width, height)
            if not detections:
                continue

            for detection in detections:
                # --- PILLAR-FIRST LOGIC ---
                # For blue/yellow we want to:
                #   1) publish a candidate even if LiDAR depth is missing (inf/NaN),
                #      so the controller can approach/center to obtain a valid depth.
                #   2) only CONFIRM/MAP when pillar is FULL + depth is valid + multi-frame passes.
                if color_name in ("blue", "yellow"):
                    score, is_full = self._evaluate_pillar_visibility(detection, width, height)

                    # pixel -> bearing (no depth needed)
                    camera_fov = self.camera.getFov()
                    norm_u = (detection.centroid_u - width / 2) / (width / 2)
                    pixel_bearing = float(norm_u * (camera_fov / 2))
                    world_bearing = float(yaw + pixel_bearing)

                    # get depth if possible (may be inf/NaN when far or occluded)
                    depth = None
                    angle_diff = None
                    if len(lidar_ranges) > 0 and len(lidar_angles) > 0:
                        angle_diffs = np.abs(lidar_angles - world_bearing)
                        closest_idx = int(np.argmin(angle_diffs))
                        angle_diff = float(angle_diffs[closest_idx])

                        if angle_diff <= self.max_angle_diff:
                            lo = max(0, closest_idx - 2)
                            hi = min(len(lidar_ranges) - 1, closest_idx + 2)
                            window = [float(lidar_ranges[i]) for i in range(lo, hi + 1)]
                            window = [d for d in window if math.isfinite(d) and d > 0.0]
                            if window:
                                dmed = float(np.median(window))
                                if math.isfinite(dmed) and dmed > 0.0:
                                    depth = dmed

                    # publish candidate (best score wins)
                    prev = self._latest_candidate[color_name]
                    if (prev is None) or (score > float(prev.get("score", 0.0))):
                        self._latest_candidate[color_name] = {
                            "t": time.time(),
                            "score": float(score),
                            "full": bool(is_full),
                            "pixel_bearing": float(pixel_bearing),
                            "world_bearing": float(world_bearing),
                            "depth": float(depth) if depth is not None else None,
                            "bbox": detection.bbox,
                            "centroid_u": float(detection.centroid_u),
                            "centroid_v": float(detection.centroid_v),
                        }

                    # Do NOT confirm/map until pillar is FULL
                    if not is_full:
                        continue

                    # For confirmation we require valid depth
                    if depth is None:
                        continue
                    if depth < self.min_lidar_range or depth > self.max_lidar_range:
                        continue

                else:
                    # Non-pillars: estimate bearing + depth first (needed for filtering)
                    bearing_depth = self._compute_bearing_and_depth(detection, yaw, lidar_ranges, lidar_angles)
                    if bearing_depth is None:
                        continue
                    pixel_bearing, world_bearing, depth, angle_diff = bearing_depth

                    # global depth validation
                    if depth < self.min_lidar_range or depth > self.max_lidar_range:
                        continue

                    # near-only gating (only applies to green/red now)
                    max_allowed = self.near_only_colors.get(detection.color, None)
                    if max_allowed is not None and depth > max_allowed:
                        continue

                # from here: world projection + occupancy verification + confirmation
                projection = self._project_to_world_with_lidar(
                    detection, robot_x, robot_z, yaw,
                    lidar_ranges, lidar_angles, occupancy_grid
                )
                if not projection:
                    continue

                world_x, world_z, depth_confidence, grid_x, grid_z = projection

                # de-duplicate
                is_duplicate = False
                for existing in self.semantic_detections[color_name]:
                    if math.hypot(existing.world_x - world_x, existing.world_z - world_z) < 0.2:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

                confirmed, final_x, final_z, confidence = self.confirmations[color_name].add_observation(
                    color_name, world_x, world_z
                )
                if confirmed:
                    final_grid_x = int(round(final_x / self.resolution)) + self.center
                    final_grid_z = self.center - int(round(final_z / self.resolution))

                    waypoint = ConfirmedWaypoint(
                        world_x=final_x,
                        world_z=final_z,
                        grid_x=final_grid_x,
                        grid_z=final_grid_z,
                        color=color_name,
                        confidence=confidence,
                        confirmed_at=time.time()
                    )
                    self.semantic_detections[color_name].append(waypoint)
                    print(f"✅ Confirmed {color_name.upper()} at ({final_x:.2f}, {final_z:.2f}), grid ({final_grid_x}, {final_grid_z})")
                    self.detection_count[color_name] += 1
                    new_confirmation = True

        return new_confirmation

    # ----------------------------
    # Detection + gating
    # ----------------------------
    def _detect_color_blobs(self, img_hsv: np.ndarray, color: str, width: int, height: int) -> List[ColorBlobDetection]:
        if color == 'red':
            mask1 = cv2.inRange(img_hsv, self.color_ranges[color]['lower1'], self.color_ranges[color]['upper1'])
            mask2 = cv2.inRange(img_hsv, self.color_ranges[color]['lower2'], self.color_ranges[color]['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(img_hsv, self.color_ranges[color]['lower'], self.color_ranges[color]['upper'])

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[ColorBlobDetection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_blob_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            # edge ignore (still OK for candidates but it removes a lot of false boxes)
            if (cx < self.edge_margin or cx > width - self.edge_margin or
                cy < self.edge_margin or cy > height - self.edge_margin):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            dets.append(ColorBlobDetection(
                centroid_u=cx,
                centroid_v=cy,
                color=color,
                color_id={'blue': 1, 'yellow': 2, 'green': 3, 'red': 4}[color],
                area=area,
                bbox=(int(x), int(y), int(w), int(h))
            ))

        return dets

    def _evaluate_pillar_visibility(self, det: ColorBlobDetection, width: int, height: int) -> Tuple[float, bool]:
        """
        Returns:
          score in [0,1] (used to decide whether controller should approach)
          is_full: True only when the pillar looks "fully visible"
        """
        x, y, w, h = det.bbox

        h_ratio = h / max(1.0, float(height))
        w_ratio = w / max(1.0, float(width))
        area_ratio = det.area / max(1.0, float(width * height))

        # centeredness in [0,1]
        centered = 1.0 - min(1.0, abs(det.centroid_u - width / 2) / (width / 2))

        bottom = (y + h) / max(1.0, float(height))

        # For close pillars, the bbox often touches the top/bottom of the image.
        # Left/right clipping is what usually means "partial glimpse".
        # FIX A: Use smaller edge margin for clipping detection
        pillar_edge = max(8, int(round(0.02 * width)))  # smaller than 40px for more lenient clipping
        clipped_lr = (
            x <= pillar_edge or
            (x + w) >= (width - pillar_edge)
        )

        # score: encourages big + tall + centered + near bottom
        score = 0.0
        score += 0.45 * min(1.0, h_ratio / self.pillar_full_h_ratio)
        score += 0.25 * min(1.0, w_ratio / self.pillar_full_w_ratio)
        score += 0.20 * centered
        score += 0.10 * min(1.0, area_ratio / 0.10)

        # "full pillar" gate (strict, but allow edge-touching for very large/close pillars)
        center_ok = abs(det.centroid_u - width / 2) <= self.pillar_center_tol * (width / 2)
        very_large = (w_ratio >= 0.35) or (h_ratio >= 0.55)  # FIX A: Allow clipping if pillar is very large (close)
        full = (
            (h_ratio >= self.pillar_full_h_ratio) and
            (w_ratio >= self.pillar_full_w_ratio) and
            (bottom >= self.pillar_bottom_ratio) and
            center_ok and
            ((not clipped_lr) or very_large)  # FIX A: Accept clipped if very large
        )

        # if it's clearly clipped, heavily downscore
        if clipped_lr:
            score *= 0.6

        return float(score), bool(full)

    def _compute_bearing_and_depth(self, det: ColorBlobDetection, robot_yaw: float, lidar_ranges: np.ndarray, lidar_angles: np.ndarray):
        # pixel -> bearing
        camera_fov = self.camera.getFov()
        width = self.camera.getWidth()
        norm_u = (det.centroid_u - width / 2) / (width / 2)
        pixel_bearing = float(norm_u * (camera_fov / 2))
        world_bearing = float(robot_yaw + pixel_bearing)

        if len(lidar_ranges) == 0 or len(lidar_angles) == 0:
            return None

        angle_diffs = np.abs(lidar_angles - world_bearing)
        closest_idx = int(np.argmin(angle_diffs))
        if float(angle_diffs[closest_idx]) > self.max_angle_diff:
            return None

        # Robust depth: median of a small window around the closest bearing
        lo = max(0, closest_idx - 2)
        hi = min(len(lidar_ranges) - 1, closest_idx + 2)
        window = [float(lidar_ranges[i]) for i in range(lo, hi + 1)]
        window = [d for d in window if math.isfinite(d) and d > 0.0]
        depth = float(np.median(window)) if window else float(lidar_ranges[closest_idx])
        return pixel_bearing, world_bearing, depth, float(angle_diffs[closest_idx])

    # ----------------------------
    # World projection (unchanged logic except kept as is)
    # ----------------------------
    def _project_to_world_with_lidar(self, detection: ColorBlobDetection,
                                     robot_x: float, robot_z: float, robot_yaw: float,
                                     lidar_ranges: np.ndarray, lidar_angles: np.ndarray,
                                     occupancy_grid: np.ndarray) -> Optional[Tuple]:
        camera_fov = self.camera.getFov()
        width = self.camera.getWidth()

        norm_u = (detection.centroid_u - width / 2) / (width / 2)
        pixel_bearing = norm_u * (camera_fov / 2)
        world_bearing = robot_yaw + pixel_bearing

        if len(lidar_ranges) == 0 or len(lidar_angles) == 0:
            return None

        angle_diffs = np.abs(lidar_angles - world_bearing)
        closest_idx = np.argmin(angle_diffs)

        if angle_diffs[closest_idx] > self.max_angle_diff:
            return None

        # Robust depth: median of a small window around the closest bearing
        lo = max(0, closest_idx - 2)
        hi = min(len(lidar_ranges) - 1, closest_idx + 2)
        window = [float(lidar_ranges[i]) for i in range(lo, hi + 1)]
        window = [d for d in window if math.isfinite(d) and d > 0.0]
        depth = float(np.median(window)) if window else float(lidar_ranges[closest_idx])

        if depth < self.min_lidar_range or depth > self.max_lidar_range:
            return None

        # near-only gating (only for colors present in near_only_colors: green/red)
        if detection.color in self.near_only_colors:
            max_allowed = self.near_only_colors[detection.color]
            if depth > max_allowed:
                return None

        # camera position
        camera_x = robot_x + self.cam_offset_forward * math.cos(robot_yaw)
        camera_z = robot_z + self.cam_offset_forward * math.sin(robot_yaw)

        # project
        world_x = camera_x + depth * math.cos(world_bearing)
        world_z = camera_z + depth * math.sin(world_bearing)

        # grid
        grid_x = int(round(world_x / self.resolution)) + self.center
        grid_z = self.center - int(round(world_z / self.resolution))

        grid_size = int(occupancy_grid.shape[0])
        if not (0 <= grid_x < grid_size and 0 <= grid_z < grid_size):
            return None

        # occupancy verification
        cell = int(occupancy_grid[grid_z, grid_x])

        if detection.color in ("red", "green"):
            if cell != 0:
                return None

        if detection.color in ("blue", "yellow"):
            # Do NOT require occupancy at the endpoint for pillars.
            # The pillar cell is often still FREE (255) due to mapping logic, so requiring (0/127) blocks confirmation.
            pass

        size_score = min(1.0, float(detection.area) / 1000.0)
        distance_score = max(0.0, 1.0 - depth / self.max_lidar_range)
        angle_score = 1.0 - (float(angle_diffs[closest_idx]) / self.max_angle_diff)
        confidence = size_score * 0.3 + distance_score * 0.4 + angle_score * 0.3

        return float(world_x), float(world_z), float(confidence), int(grid_x), int(grid_z)

    # ----------------------------
    # Visualization helper
    # ----------------------------
    def draw_detections_on_image(self, image_bgr: np.ndarray) -> np.ndarray:
        result = image_bgr.copy()
        height, width = result.shape[:2]
        img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        box_colors = {'blue': (255, 0, 0), 'yellow': (0, 255, 255), 'green': (0, 255, 0), 'red': (0, 0, 255)}

        for color_name in ['blue', 'yellow', 'green', 'red']:
            if color_name == 'red':
                mask1 = cv2.inRange(img_hsv, self.color_ranges[color_name]['lower1'], self.color_ranges[color_name]['upper1'])
                mask2 = cv2.inRange(img_hsv, self.color_ranges[color_name]['lower2'], self.color_ranges[color_name]['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(img_hsv, self.color_ranges[color_name]['lower'], self.color_ranges[color_name]['upper'])

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_blob_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                color_bgr = box_colors[color_name]
                has_det = self.has_waypoint(color_name)
                thickness = 3 if has_det else 2
                cv2.rectangle(result, (x, y), (x + w, y + h), color_bgr, thickness)

                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 6, color_bgr, -1)

                label = f"{color_name.upper()}"
                if has_det:
                    label += f" ({len(self.semantic_detections[color_name])})"
                else:
                    # show FULL/CAND for pillars
                    if color_name in ("blue", "yellow"):
                        cand = self.get_candidate(color_name)
                        if cand and cand.get("full", False):
                            label += " [FULL]"
                        elif cand and float(cand.get("score", 0.0)) >= self.pillar_candidate_score_min:
                            label += " [CAND]"
                cv2.putText(result, label, (x, max(10, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

        return result