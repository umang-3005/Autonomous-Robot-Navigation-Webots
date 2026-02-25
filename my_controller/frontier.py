import numpy as np
import cv2


class FrontierAnalyzer:
    """Compute and display frontier lines on a `LidarMap` grid."""

    def __init__(self, lidar_map, *, min_frontier_cells: int = 2, seed: int = 42):
        self.map = lidar_map
        self.min_frontier_cells = int(min_frontier_cells)
        self.rng = np.random.default_rng(seed=seed)
        self.frontier_centroids = set()  # list of (x, y) tuples for frontier centroids

    def create_frontier_mask(self):
        """Return a binary (0/255) frontier mask derived from map data."""
        grid = self.map.grid_map
        unknown = (grid == 127).astype(np.uint8)
        free = (grid == 255).astype(np.uint8)
        dilated = cv2.dilate(unknown, np.ones((3, 3), np.uint8), iterations=1)
        return (free & dilated).astype(np.uint8) * 255

    def extract_frontiers(self):
        """
        Returns a list of (grid_x, grid_z) tuples representing the centroids
        of frontier blobs (free cells adjacent to unknown).
        """
        mask = self.create_frontier_mask()

        # Filter small noise
        n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        frontier_points = []
        for i in range(1, n_lbl):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.min_frontier_cells:
                cx, cy = centroids[i]
                frontier_points.append((int(round(cx)), int(round(cy))))
        return frontier_points

    def visualize_frontier_lines(self, win_name="Colored Frontier Lines"):
        """
        Show the occupancy grid with:
        
        - Yellow 1-px frontier contours
        - A brown 3×3 pixel square centered on each contour’s centroid
        """
        mask = self.create_frontier_mask()

        # --- filter small blobs ----------------------------------------
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered = np.zeros_like(mask)
        for i in range(1, n_lbl):  # skip background 0
            if stats[i, cv2.CC_STAT_AREA] >= self.min_frontier_cells:
                filtered[labels == i] = 255

        contours, _ = cv2.findContours(
            filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Visualization -------------------------------------------------
        vis = cv2.cvtColor(self.map.grid_map.copy(), cv2.COLOR_GRAY2BGR)
        self.map._draw_robot_on(vis)

        yellow = (0, 255, 255)       # BGR yellow
        brown = (19, 69, 139)        # BGR brown (rust-like tone)

        for c in contours:
            # draw the contour
            cv2.drawContours(vis, [c], -1, yellow, 1)

            # compute centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:  # fall-back to bounding-box center
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
            self.frontier_centroids.add((cx, cy))
            # draw a 3×3 brown square at the centroid
            cv2.rectangle(vis, (cx - 1, cy - 1), (cx + 1, cy + 1), brown, thickness=-1)

        cv2.imshow(win_name, vis)
        cv2.waitKey(1)