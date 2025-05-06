import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class RobotColor(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    YELLOW = 4
    BLACK = 5  # or WHITE, depending on the game version

class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class BoardDetector:
    """
    Detects and processes the Ricochet Robots board from an image.
    """

    # Color ranges (HSV) for each robot
    ROBOT_COLOR_RANGES = {
        RobotColor.RED: ((0, 100, 100), (10, 255, 255)),
        RobotColor.GREEN: ((35, 70, 70), (85, 255, 255)),
        RobotColor.BLUE: ((100, 80, 80), (140, 255, 255)),
        RobotColor.YELLOW: ((20, 100, 100), (35, 255, 255)),
        RobotColor.BLACK: ((0, 0, 0), (180, 30, 50)),  # Very dark colors
    }

    # Color ranges for target symbols
    TARGET_COLOR_RANGES = {
        RobotColor.RED: ((0, 70, 70), (10, 255, 255)),
        RobotColor.GREEN: ((35, 70, 70), (85, 255, 255)),
        RobotColor.BLUE: ((100, 80, 80), (140, 255, 255)),
        RobotColor.YELLOW: ((20, 100, 100), (35, 255, 255)),
    }

    def __init__(self, image_path: str):
        """
        Initialize the board detector with the path to the image.

        Args:
            image_path: Path to the input image
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Resize image if it's too large for better processing
        max_size = 1200
        h, w = self.original_image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            self.original_image = cv2.resize(self.original_image, (int(w * scale), int(h * scale)))

        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

        # Debug images for visualization - moved this before _preprocess_image() call
        self.debug_images = {}

        # Apply preprocessing to improve feature detection
        self.preprocessed = self._preprocess_image()

        # Board dimensions - will be detected
        self.board_size = (16, 16)  # Default size for Ricochet Robots
        self.grid_cells = []
        self.walls = []
        self.robots = {}
        self.targets = {}
        self.active_target = None

    def _preprocess_image(self) -> np.ndarray:
        """
        Preprocess the image for better feature detection.

        Returns:
            Preprocessed image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Perform morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        self.debug_images["preprocessed"] = cleaned
        return cleaned

    def process(self) -> bool:
        """
        Process the image to extract board information.

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # Detect board grid
        if not self._detect_grid():
            return False

        # Detect walls
        self._detect_walls()

        # Detect robots
        self._detect_robots()

        # Detect targets
        self._detect_targets()

        # If no active target was specified, pick the first one
        if self.active_target is None and self.targets:
            self.active_target = next(iter(self.targets.keys()))

        return True

    def _detect_grid(self) -> bool:
        """
        Detect the grid cells of the board.

        Returns:
            bool: True if grid detection succeeded, False otherwise
        """
        # For Ricochet Robots, we know the board is 16x16
        # So we'll create a uniform grid based on the image dimensions
        height, width = self.original_image.shape[:2]

        # Try to find board corners or boundaries
        # For simplicity, we'll assume the board takes the whole image
        # and create a 16x16 grid

        # Calculate cell size
        cell_width = width / self.board_size[0]
        cell_height = height / self.board_size[1]

        # Create grid cells
        self.grid_cells = []
        for y in range(self.board_size[1]):
            row = []
            for x in range(self.board_size[0]):
                top_left = (int(x * cell_width), int(y * cell_height))
                bottom_right = (int((x + 1) * cell_width), int((y + 1) * cell_height))
                center = (int(top_left[0] + cell_width / 2), int(top_left[1] + cell_height / 2))
                row.append({
                    "position": (x, y),
                    "bbox": (top_left, bottom_right),
                    "center": center
                })
            self.grid_cells.append(row)

        # Create a debug image showing the grid
        grid_debug = self.original_image.copy()
        for row in self.grid_cells:
            for cell in row:
                center = cell["center"]
                position = cell["position"]
                (top_left, bottom_right) = cell["bbox"]

                # Draw cell rectangle
                cv2.rectangle(grid_debug, top_left, bottom_right, (200, 200, 200), 1)

                # Add coordinates for some cells to avoid clutter
                if position[0] % 4 == 0 and position[1] % 4 == 0:
                    cv2.putText(grid_debug, f"{position[0]},{position[1]}",
                               (center[0]-20, center[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        self.debug_images["grid"] = grid_debug
        return True

    def _detect_walls(self) -> None:
        """
        Detect walls on the board using edge detection and line detection.
        """
        # For a real implementation, we'd use line detection algorithms
        # For now, let's use a simplified approach to detect horizontal and vertical lines

        # Start with detecting edges
        edges = cv2.Canny(self.gray_image, 50, 150)

        # Apply Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

        # Create a debug image to show detected lines
        line_debug = self.original_image.copy()

        # Create an empty list for walls
        horizontal_walls = []
        vertical_walls = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Determine if the line is horizontal or vertical
                if abs(y2 - y1) < abs(x2 - x1) * 0.2:  # Horizontal line
                    cv2.line(line_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    horizontal_walls.append((x1, y1, x2, y2))
                elif abs(x2 - x1) < abs(y2 - y1) * 0.2:  # Vertical line
                    cv2.line(line_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    vertical_walls.append((x1, y1, x2, y2))

        self.debug_images["lines"] = line_debug

        # Convert detected lines to wall segments between grid cells
        # This is a simplified approach - for real-world usage, more robust methods would be needed

        # For now, let's just use the outer border as walls for demonstration
        # In a real implementation, we would map the detected lines to grid cells

        # Add outer border walls
        for y in range(self.board_size[1] + 1):
            for x in range(self.board_size[0]):
                # Add horizontal walls on the top and bottom borders
                if y == 0 or y == self.board_size[1]:
                    self.walls.append(((x, y), (x+1, y)))

        for x in range(self.board_size[0] + 1):
            for y in range(self.board_size[1]):
                # Add vertical walls on the left and right borders
                if x == 0 or x == self.board_size[0]:
                    self.walls.append(((x, y), (x, y+1)))

        # Simulate some inner walls - in a real implementation, these would be detected from the image
        # Add some horizontal and vertical walls to simulate the board
        # (These would be determined by analyzing the detected lines in a real implementation)
        for i in range(1, self.board_size[0], 4):
            for j in range(1, self.board_size[1], 4):
                # Add some horizontal and vertical walls to simulate internal walls
                self.walls.append(((i, j), (i+2, j)))
                self.walls.append(((i, j), (i, j+2)))

    def _detect_robots(self) -> None:
        """
        Detect robots on the board using color thresholding.
        """
        robot_debug = self.original_image.copy()

        for color, (lower, upper) in self.ROBOT_COLOR_RANGES.items():
            # Create mask for the current color
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(self.hsv_image, lower, upper)

            # Perform morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area to find the robots
            robot_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Adjust this threshold based on your image size
                    robot_contours.append(contour)

            # For each potential robot, find its position on the grid
            for contour in robot_contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw the contour and center in the debug image
                    cv2.drawContours(robot_debug, [contour], 0, (0, 255, 0), 2)
                    cv2.circle(robot_debug, (cx, cy), 5, (0, 0, 255), -1)

                    # Find the closest grid cell
                    closest_cell = self._find_closest_cell((cx, cy))
                    if closest_cell:
                        self.robots[color] = closest_cell

                        # Draw the grid cell in the debug image
                        grid_cell = self.grid_cells[closest_cell[1]][closest_cell[0]]
                        (top_left, bottom_right) = grid_cell["bbox"]
                        cv2.rectangle(robot_debug, top_left, bottom_right, (255, 0, 0), 2)

                        # Add a text label
                        color_name = color.name if hasattr(color, 'name') else 'Unknown'
                        cv2.putText(robot_debug, color_name, (cx, cy - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.debug_images["robots"] = robot_debug

    def _detect_targets(self) -> None:
        """
        Detect target symbols on the board using color and shape detection.
        """
        target_debug = self.original_image.copy()

        # Loop through each target color
        for color, (lower, upper) in self.TARGET_COLOR_RANGES.items():
            # Create mask for the current color
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(self.hsv_image, lower, upper)

            # Perform morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area to find potential targets
            target_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                # Targets are usually smaller than robots
                if 20 < area < 200:  # Adjust thresholds based on your images
                    target_contours.append(contour)

            # For each potential target, find its position on the grid
            for contour in target_contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Check if the potential target is not too close to a robot
                    too_close_to_robot = False
                    for robot_color, robot_pos in self.robots.items():
                        robot_cell = self.grid_cells[robot_pos[1]][robot_pos[0]]
                        robot_center = robot_cell["center"]
                        dist = ((cx - robot_center[0])**2 + (cy - robot_center[1])**2)**0.5
                        if dist < 20:  # Adjust this threshold
                            too_close_to_robot = True
                            break

                    if too_close_to_robot:
                        continue

                    # Draw the contour and center in the debug image
                    cv2.drawContours(target_debug, [contour], 0, (0, 255, 255), 2)
                    cv2.circle(target_debug, (cx, cy), 3, (255, 0, 0), -1)

                    # Find the closest grid cell
                    closest_cell = self._find_closest_cell((cx, cy))
                    if closest_cell:
                        # Check if this cell is already assigned to a target
                        cell_already_has_target = False
                        for existing_color, existing_pos in self.targets.items():
                            if existing_pos == closest_cell:
                                cell_already_has_target = True
                                break

                        if not cell_already_has_target:
                            self.targets[color] = closest_cell

                            # Draw the grid cell in the debug image
                            grid_cell = self.grid_cells[closest_cell[1]][closest_cell[0]]
                            (top_left, bottom_right) = grid_cell["bbox"]
                            cv2.rectangle(target_debug, top_left, bottom_right, (0, 255, 255), 2)

                            # Add a text label
                            color_name = color.name if hasattr(color, 'name') else 'Unknown'
                            cv2.putText(target_debug, color_name, (cx, cy - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        self.debug_images["targets"] = target_debug

        # For demonstration, set the active target if targets were found
        if self.targets:
            # In a real implementation, this would be detected based on specific markings or rules
            self.active_target = next(iter(self.targets.keys()))

    def _find_closest_cell(self, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find the closest grid cell to a given point.

        Args:
            point: (x, y) coordinates

        Returns:
            Tuple containing the grid coordinates of the closest cell, or None if no cells
        """
        if not self.grid_cells:
            return None

        min_dist = float('inf')
        closest_cell = None

        for row in self.grid_cells:
            for cell in row:
                center = cell["center"]
                dist = (point[0] - center[0])**2 + (point[1] - center[1])**2

                if dist < min_dist:
                    min_dist = dist
                    closest_cell = cell["position"]

        return closest_cell

    def get_board_state(self) -> Dict:
        """
        Return the complete board state.

        Returns:
            Dict containing board state information
        """
        return {
            "board_size": self.board_size,
            "walls": self.walls,
            "robots": self.robots,
            "targets": self.targets,
            "active_target": self.active_target
        }