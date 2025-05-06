import cv2
import numpy as np
from typing import Dict, List, Tuple
from src.image_processing.board_detector import RobotColor

class PathVisualizer:
    """
    Visualizes the solution path on the board image.
    """

    # Color mappings for each robot
    COLORS = {
        RobotColor.RED: (0, 0, 255),  # BGR format
        RobotColor.GREEN: (0, 255, 0),
        RobotColor.BLUE: (255, 0, 0),
        RobotColor.YELLOW: (0, 255, 255),
        RobotColor.BLACK: (0, 0, 0)
    }

    def __init__(self, original_image: np.ndarray, grid_cells: List[List[Dict]]):
        """
        Initialize the visualizer with the original image and grid cells.

        Args:
            original_image: Original board image
            grid_cells: Detected grid cells with positions
        """
        self.original_image = original_image.copy()
        self.grid_cells = grid_cells
        self.cell_size = self._calculate_cell_size()

    def _calculate_cell_size(self) -> int:
        """
        Calculate the size of a grid cell.

        Returns:
            Size of a grid cell in pixels
        """
        # For simplicity, assume all cells are the same size
        if self.grid_cells and self.grid_cells[0]:
            cell = self.grid_cells[0][0]
            (x1, y1), (x2, y2) = cell["bbox"]
            return max(x2 - x1, y2 - y1)
        return 30  # Default size if grid detection failed

    def draw_solution_path(self, paths: List[Tuple[RobotColor, List[Tuple[int, int]]]]) -> np.ndarray:
        """
        Draw the solution path on the board image.

        Args:
            paths: List of (robot, path) tuples

        Returns:
            Image with solution path drawn
        """
        result_image = self.original_image.copy()

        # Draw each robot's path
        for robot, path in paths:
            color = self.COLORS.get(robot, (255, 255, 255))

            # Draw path as a line
            for i in range(len(path) - 1):
                start_pos = path[i]
                end_pos = path[i + 1]

                start_pixel = self._grid_to_pixel(start_pos)
                end_pixel = self._grid_to_pixel(end_pos)

                cv2.line(result_image, start_pixel, end_pixel, color, thickness=3)

                # Draw an arrow at every step
                self._draw_arrow(result_image, start_pixel, end_pixel, color)

            # Mark the final position
            if path:
                final_pos = path[-1]
                final_pixel = self._grid_to_pixel(final_pos)
                cv2.circle(result_image, final_pixel, 10, color, -1)

        # Add text with move count
        total_moves = sum(len(path) - 1 for _, path in paths if path)
        cv2.putText(
            result_image,
            f"Total moves: {total_moves}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        return result_image

    def _grid_to_pixel(self, grid_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert grid coordinates to pixel coordinates.

        Args:
            grid_pos: Grid position (x, y)

        Returns:
            Pixel coordinates (x, y)
        """
        if self.grid_cells and len(self.grid_cells) > grid_pos[1] and len(self.grid_cells[grid_pos[1]]) > grid_pos[0]:
            # If we have detected the grid cells, use their centers
            cell = self.grid_cells[grid_pos[1]][grid_pos[0]]
            return cell["center"]

        # Fallback to simple calculation
        x = int((grid_pos[0] + 0.5) * self.cell_size)
        y = int((grid_pos[1] + 0.5) * self.cell_size)
        return (x, y)

    def _draw_arrow(self, image: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        """
        Draw an arrow on the image.

        Args:
            image: Image to draw on
            start: Start point (x, y)
            end: End point (x, y)
            color: Arrow color (B, G, R)
        """
        # Calculate angle of the line
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])

        # Calculate arrow head points
        arrow_length = 20
        arrow_angle = np.pi / 6  # 30 degrees

        # Calculate position for the arrow head
        # Place it at 70% of the way from start to end
        pos_x = int(start[0] + 0.7 * (end[0] - start[0]))
        pos_y = int(start[1] + 0.7 * (end[1] - start[1]))
        pos = (pos_x, pos_y)

        # Calculate arrow head points
        p1_x = int(pos[0] - arrow_length * np.cos(angle + arrow_angle))
        p1_y = int(pos[1] - arrow_length * np.sin(angle + arrow_angle))
        p2_x = int(pos[0] - arrow_length * np.cos(angle - arrow_angle))
        p2_y = int(pos[1] - arrow_length * np.sin(angle - arrow_angle))

        # Draw arrow head
        cv2.line(image, pos, (p1_x, p1_y), color, 2)
        cv2.line(image, pos, (p2_x, p2_y), color, 2)

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save the image to a file.

        Args:
            image: Image to save
            output_path: Path to save the image to
        """
        cv2.imwrite(output_path, image)