# cost_aware_planner.py
"""
A* Path Planner with Semantic Cost Awareness
- Handles semantic costs (avoid green poison, penalize red walls)
- 8-connected grid movement
- Ramer-Douglas-Peucker path simplification
"""

import numpy as np
import heapq
import math
import cv2

def create_cost_map(
    occupancy_grid,
    unknown_as_obstacle=True,
    inflation_radius=4,
    poison_area_mask=None,
    poison_penalty=1e6,
    red_wall_area_mask=None,
    red_wall_penalty=20.0,
):
    """
    Build a cost map for A* from occupancy + semantic masks
    """

    height, width = occupancy_grid.shape

    # Start with normal cost
    cost_map = np.ones((height, width), dtype=np.float32)

    # Obstacles
    cost_map[occupancy_grid == 0] = np.inf

    if unknown_as_obstacle:
        cost_map[occupancy_grid == 127] = np.inf

    # Inflate obstacles (remove too-narrow passages)
    if inflation_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_radius + 1, 2 * inflation_radius + 1)
        )
        obstacles = np.isinf(cost_map).astype(np.uint8)
        obstacles = cv2.dilate(obstacles, kernel)
        cost_map[obstacles > 0] = np.inf

    # Green poison = forbidden
    if poison_area_mask is not None:
        cost_map[poison_area_mask] = poison_penalty

    # Red walls = avoid but not blocked
    if red_wall_area_mask is not None:
        cost_map[red_wall_area_mask] += red_wall_penalty

    return cost_map

# 8-connected movement directions
MOVEMENT_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]


def calculate_octile_heuristic(point_a, point_b):
    """
    Octile distance heuristic for 8-connected grids
    Admissible for diagonal movement with cost sqrt(2)
    """
    dx, dy = abs(point_a[0] - point_b[0]), abs(point_a[1] - point_b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


class SemanticPathPlanner:
    """
    A* planner with semantic cost integration
    """

    def __init__(self, cost_map):
        """
        Initialize planner with cost map

        Args:
            cost_map: 2D numpy array where:
                - 1.0 = free space (normal cost)
                - np.inf = obstacle (impassable)
                - >1.0 = high cost areas (e.g., near poison)
        """
        self.cost_map = cost_map
        self.path = []

    def find_path(self, start, goal):
        """
        Plan path from start to goal using A*

        Args:
            start: (x, y) tuple - start grid coordinates
            goal: (x, y) tuple - goal grid coordinates

        Returns:
            List of (x, y) waypoints from start to goal, or [] if no path exists
        """
        height, width = self.cost_map.shape

        # Validate start and goal
        if not self._is_within_bounds(start, width, height) or not self._is_within_bounds(goal, width, height):
            print(f"⚠️  Start {start} or Goal {goal} outside grid bounds ({width}x{height})")
            return []

        # Check if start/goal are on obstacles
        if self.cost_map[start[1], start[0]] == np.inf:
            print(f"⚠️  Start {start} is on an obstacle!")
            return []
        if self.cost_map[goal[1], goal[0]] == np.inf:
            print(f"⚠️  Goal {goal} is on an obstacle!")
            return []

        # A* search
        frontier = [(0.0, start)]
        came_from = {start: None}
        g_score = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)

            # Goal reached
            if current == goal:
                break

            cx, cy = current

            # Explore neighbors
            for dx, dy in MOVEMENT_DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                # Skip if out of bounds
                if not self._is_within_bounds(neighbor, width, height):
                    continue

                # --- prevent diagonal corner-cutting ---
                if dx != 0 and dy != 0:
                    # if either adjacent cardinal cell is blocked, don't allow diagonal move
                    if self.cost_map[cy, cx + dx] == np.inf or self.cost_map[cy + dy, cx] == np.inf:
                        continue

                # Skip if obstacle
                cell_cost = self.cost_map[ny, nx]
                if cell_cost == np.inf:
                    continue

                # Calculate movement cost
                step_cost = math.hypot(dx, dy)  # sqrt(2) for diagonal, 1.0 for cardinal
                tentative_g = g_score[current] + step_cost + float(cell_cost)

                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + calculate_octile_heuristic(neighbor, goal)
                    heapq.heappush(frontier, (f_score, neighbor))
                    came_from[neighbor] = current

        # Check if goal was reached
        if goal not in came_from:
            print(f"⚠️  No path found from {start} to {goal}")
            return []

        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        self.path = path

        return path

    def simplify_path(self, epsilon=2.0):
        """
        Simplify path using Ramer-Douglas-Peucker algorithm
        Reduces waypoint count while maintaining path shape

        Args:
            epsilon: Maximum perpendicular distance threshold (in grid cells)
                     Higher = more aggressive simplification

        Returns:
            Simplified path (also stored in self.path)
        """
        if len(self.path) < 3:
            return self.path

        def perpendicular_distance(point, line_start, line_end):
            """
            Calculate perpendicular distance from point to line segment
            """
            ax, ay = line_start
            bx, by = line_end
            px, py = point

            # Vector from A to B
            vx, vy = bx - ax, by - ay

            # Handle degenerate case (A == B)
            if vx == 0 and vy == 0:
                return math.hypot(px - ax, py - ay)

            # Project point onto line (clamped to segment)
            t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            qx, qy = ax + t * vx, ay + t * vy

            return math.hypot(px - qx, py - qy)

        def rdp_recursive(points):
            """
            Recursive Ramer-Douglas-Peucker implementation
            """
            if len(points) < 3:
                return points

            # Find point with maximum distance from line
            start, end = points[0], points[-1]
            max_idx, max_dist = 0, 0.0

            for i in range(1, len(points) - 1):
                dist = perpendicular_distance(points[i], start, end)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            # If max distance exceeds threshold, split recursively
            if max_dist > epsilon:
                left_segment = rdp_recursive(points[:max_idx + 1])
                right_segment = rdp_recursive(points[max_idx:])
                return left_segment[:-1] + right_segment  # Avoid duplicate at split point
            else:
                # All points within threshold - keep only endpoints
                return [start, end]

        # Apply RDP algorithm
        self.path = rdp_recursive(self.path)

        return self.path

    def _is_within_bounds(self, point, width, height):
        """Check if point is inside grid bounds"""
        x, y = point
        return 0 <= x < width and 0 <= y < height

    def calculate_path_length(self):
        """
        Calculate total path length in grid cells

        Returns:
            Total Euclidean distance along path
        """
        if len(self.path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            total_length += math.hypot(x2 - x1, y2 - y1)
        return total_length

    def display_path(self, occupancy_grid, window_name="Planned Path"):
        """
        Visualize the planned path on the occupancy grid

        Args:
            occupancy_grid: Grayscale occupancy grid (0=obstacle, 255=free, 127=unknown)
            window_name: OpenCV window name
        """
        if not self.path:
            print("⚠️  No path to visualize")
            return

        # Convert to BGR for colored visualization
        visualization = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2BGR)

        # Draw path
        for i in range(len(self.path) - 1):
            cv2.line(visualization, self.path[i], self.path[i + 1], (0, 165, 255), 2)

        # Draw waypoints
        for point in self.path:
            cv2.circle(visualization, point, 2, (255, 255, 255), -1)

        # Highlight start and goal
        if len(self.path) >= 2:
            cv2.circle(visualization, self.path[0], 8, (0, 255, 0), -1)   # Green start
            cv2.circle(visualization, self.path[-1], 8, (0, 255, 255), -1)  # Yellow goal

        cv2.imshow(window_name, visualization)
        cv2.waitKey(1)


# ============================================================================
# TESTING / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test the planner with a simple scenario"""
    
    # Create a 100x100 test grid
    grid_size = 100
    cost_map = np.ones((grid_size, grid_size), dtype=np.float32)
    
    # Add some obstacles
    cost_map[30:70, 45:55] = np.inf  # Vertical wall
    
    # Add high-cost area (semantic: poison field)
    cost_map[40:60, 60:80] = 50.0
    
    # Plan path
    planner = SemanticPathPlanner(cost_map)
    
    start = (10, 50)
    goal = (90, 50)
    
    print(f"Planning from {start} to {goal}...")
    path = planner.find_path(start, goal)
    
    if path:
        print(f"✓ Path found: {len(path)} waypoints")
        print(f"  Path length: {planner.calculate_path_length():.2f} cells")
        
        # Simplify
        planner.simplify_path(epsilon=3.0)
        print(f"✓ Simplified: {len(planner.path)} waypoints")
        print(f"  Path length: {planner.calculate_path_length():.2f} cells")
        
        # Visualize
        occupancy = (cost_map != np.inf).astype(np.uint8) * 255
        planner.display_path(occupancy)
        cv2.waitKey(0)
    else:
        print("❌ No path found")
