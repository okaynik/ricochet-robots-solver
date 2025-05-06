import argparse
import os
import cv2
import numpy as np
import time
import traceback
import json
from typing import Tuple, List, Dict, Optional

from src.image_processing.board_detector import BoardDetector, RobotColor
from src.solver.path_finder import PathFinder
from src.visualization.path_visualizer import PathVisualizer

def load_board_config(config_path: str) -> Optional[Dict]:
    """
    Load board configuration from a JSON file

    Args:
        config_path: Path to the configuration file

    Returns:
        Board configuration dictionary or None if loading failed
    """
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Convert strings back to Enum values
        robots = {}
        for k, v in data["robots"].items():
            try:
                color = RobotColor[k] if k in [e.name for e in RobotColor] else k
                robots[color] = tuple(v)
            except (KeyError, ValueError):
                robots[k] = tuple(v)

        targets = {}
        for k, v in data["targets"].items():
            try:
                color = RobotColor[k] if k in [e.name for e in RobotColor] else k
                targets[color] = tuple(v)
            except (KeyError, ValueError):
                targets[k] = tuple(v)

        active_target = None
        if data["active_target"]:
            try:
                active_target = RobotColor[data["active_target"]] if data["active_target"] in [e.name for e in RobotColor] else data["active_target"]
            except (KeyError, ValueError):
                active_target = data["active_target"]

        # Convert wall list to the expected format
        walls = [tuple(tuple(p) for p in wall) for wall in data["walls"]]

        return {
            "board_size": tuple(data["board_size"]),
            "walls": walls,
            "robots": robots,
            "targets": targets,
            "active_target": active_target
        }
    except Exception as e:
        print(f"Error loading board configuration: {str(e)}")
        traceback.print_exc()
        return None

def update_detector_with_config(detector: BoardDetector, config: Dict) -> None:
    """
    Update a BoardDetector instance with configuration data

    Args:
        detector: BoardDetector instance to update
        config: Configuration dictionary
    """
    detector.board_size = config["board_size"]
    detector.walls = config["walls"]
    detector.robots = config["robots"]
    detector.targets = config["targets"]
    detector.active_target = config["active_target"]

def process_image(input_path: str, output_path: str, debug: bool = False, config_path: str = None) -> bool:
    """
    Process a Ricochet Robots board image, find the optimal solution, and visualize it.

    Args:
        input_path: Path to the input image
        output_path: Path to save the output image
        debug: Enable debug output and images
        config_path: Path to manual configuration file (optional)

    Returns:
        True if processing succeeded, False otherwise
    """
    print(f"Processing image: {input_path}")

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        return False

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create debug directory if debug is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(os.path.dirname(output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

    try:
        # 1. Detect board
        print("Detecting board...")
        board_detector = BoardDetector(input_path)

        # If a config file is provided, use it instead of detecting
        manual_config = None
        if config_path:
            print(f"Loading configuration from {config_path}...")
            manual_config = load_board_config(config_path)
            if not manual_config:
                print("Failed to load configuration, falling back to detection")

        # Always process the image to get grid cells
        if not board_detector.process():
            print("Failed to detect board")
            return False

        # If manual config was loaded, update the detector
        if manual_config:
            print("Applying manual configuration...")
            update_detector_with_config(board_detector, manual_config)

        # If debug is enabled, save debug images
        if debug:
            debug_images = board_detector.debug_images

            # Save preprocessed image
            if "preprocessed" in debug_images:
                cv2.imwrite(os.path.join(debug_dir, "preprocessed.jpg"), debug_images["preprocessed"])

            # Save grid visualization
            if "grid" in debug_images:
                cv2.imwrite(os.path.join(debug_dir, "grid.jpg"), debug_images["grid"])

            # Save robot detection
            if "robots" in debug_images:
                cv2.imwrite(os.path.join(debug_dir, "robots.jpg"), debug_images["robots"])

            # Save target detection
            if "targets" in debug_images:
                cv2.imwrite(os.path.join(debug_dir, "targets.jpg"), debug_images["targets"])

            # Save lines detection
            if "lines" in debug_images:
                cv2.imwrite(os.path.join(debug_dir, "lines.jpg"), debug_images["lines"])

        # Generate a text representation for debugging
        if debug:
            text_representation = generate_text_representation(board_detector)
            text_output = os.path.join(debug_dir, "board_text.txt")
            with open(text_output, "w") as f:
                f.write(text_representation)
            print(f"Text representation saved to: {text_output}")

            # Create a debug visualization
            debug_image = create_debug_visualization(board_detector)
            debug_output = os.path.join(debug_dir, "board_debug.jpg")
            cv2.imwrite(debug_output, debug_image)
            print(f"Debug visualization saved to: {debug_output}")

        # Get board state
        board_state = board_detector.get_board_state()
        print(f"Detected robots: {len(board_state['robots'])}")
        print(f"Detected targets: {len(board_state['targets'])}")
        print(f"Detected walls: {len(board_state['walls'])}")

        # Check if any robots were detected
        if not board_state["robots"]:
            print("No robots detected. Cannot proceed.")
            return False

        # Check if any targets were detected
        if not board_state["targets"]:
            print("No targets detected. Cannot proceed.")
            return False

        # Check if active target was detected
        if not board_state["active_target"]:
            print("No active target detected. Cannot proceed.")
            return False

        # 2. Find solution
        print("Finding solution...")
        path_finder = PathFinder(board_state)
        solution = path_finder.find_solution()

        if solution is None:
            print("No solution found")
            return False

        print(f"Solution found with {len(solution)} moves")
        print(f"Solution path: {solution}")

        # Convert solution to paths
        paths = path_finder.get_path_coordinates(solution)

        # 3. Visualize solution
        print("Visualizing solution...")
        visualizer = PathVisualizer(board_detector.original_image, board_detector.grid_cells)
        result_image = visualizer.draw_solution_path(paths)

        # Save result
        visualizer.save_image(result_image, output_path)
        print(f"Solution image saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        traceback.print_exc()
        return False

def generate_text_representation(detector: BoardDetector) -> str:
    """
    Generate a text representation of the board

    Args:
        detector: BoardDetector instance

    Returns:
        Text representation of the board
    """
    board_size = detector.board_size
    walls = detector.walls
    robots = detector.robots
    targets = detector.targets
    active_target = detector.active_target

    # Create a grid for the text representation
    grid_width = board_size[0] * 2 + 1
    grid_height = board_size[1] * 2 + 1
    text_grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    # Fill in the grid with dots
    for y in range(board_size[1]):
        for x in range(board_size[0]):
            center_y = y * 2 + 1
            center_x = x * 2 + 1
            text_grid[center_y][center_x] = '·'

    # Draw grid lines
    for y in range(grid_height):
        for x in range(grid_width):
            if x % 2 == 0 and y % 2 == 0:
                text_grid[y][x] = '+'
            elif x % 2 == 0:
                text_grid[y][x] = '|'
            elif y % 2 == 0:
                text_grid[y][x] = '-'

    # Add walls
    for (x1, y1), (x2, y2) in walls:
        if y1 == y2:  # Horizontal wall
            wall_y = y1 * 2
            start_x = min(x1, x2) * 2
            end_x = max(x1, x2) * 2
            for x in range(start_x, end_x + 1):
                if 0 <= wall_y < grid_height and 0 <= x < grid_width:
                    text_grid[wall_y][x] = '='
        else:  # Vertical wall
            wall_x = x1 * 2
            start_y = min(y1, y2) * 2
            end_y = max(y1, y2) * 2
            for y in range(start_y, end_y + 1):
                if 0 <= wall_x < grid_width and 0 <= y < grid_height:
                    text_grid[y][wall_x] = '#'

    # Add robots
    robot_symbols = {
        RobotColor.RED: 'R',
        RobotColor.GREEN: 'G',
        RobotColor.BLUE: 'B',
        RobotColor.YELLOW: 'Y',
        RobotColor.BLACK: 'K'
    }
    for color, pos in robots.items():
        x, y = pos
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            text_grid[y * 2 + 1][x * 2 + 1] = robot_symbols.get(color, '?')

    # Add targets
    target_symbols = {
        RobotColor.RED: 'r',
        RobotColor.GREEN: 'g',
        RobotColor.BLUE: 'b',
        RobotColor.YELLOW: 'y',
        RobotColor.BLACK: 'k'
    }
    for color, pos in targets.items():
        x, y = pos
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            if text_grid[y * 2 + 1][x * 2 + 1] == '·':
                text_grid[y * 2 + 1][x * 2 + 1] = target_symbols.get(color, '?')
            else:
                text_grid[y * 2 + 1][x * 2 + 1] = f'({text_grid[y * 2 + 1][x * 2 + 1]})'

    # Highlight active target
    if active_target and active_target in targets:
        x, y = targets[active_target]
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            symbol = text_grid[y * 2 + 1][x * 2 + 1]
            if len(symbol) == 1:
                text_grid[y * 2 + 1][x * 2 + 1] = f'*{symbol}*'

    # Convert to string
    text_repr = '\n'.join([''.join(row) for row in text_grid])

    # Add summary
    summary = "\n\nBoard Summary:\n"
    summary += f"Board size: {board_size[0]}x{board_size[1]}\n"
    summary += f"Robots: {len(robots)}\n"
    robot_positions = ', '.join([f"{robot_symbols.get(color, '?')}:({pos[0]},{pos[1]})" for color, pos in robots.items()])
    summary += f"Robot positions: {robot_positions}\n"
    summary += f"Targets: {len(targets)}\n"
    target_positions = ', '.join([f"{target_symbols.get(color, '?')}:({pos[0]},{pos[1]})" for color, pos in targets.items()])
    summary += f"Target positions: {target_positions}\n"
    if active_target:
        summary += f"Active target: {target_symbols.get(active_target, '?')}\n"
    summary += f"Walls: {len(walls)}\n"

    return text_repr + summary

def create_debug_visualization(detector: BoardDetector) -> np.ndarray:
    """
    Create a debug visualization of the board

    Args:
        detector: BoardDetector instance

    Returns:
        Debug image
    """
    image = detector.original_image.copy()

    # Draw grid
    for row in detector.grid_cells:
        for cell in row:
            center = cell["center"]
            position = cell["position"]
            (top_left, bottom_right) = cell["bbox"]
            cv2.rectangle(image, top_left, bottom_right, (200, 200, 200), 1)

            # Add coordinates for some cells
            if position[0] % 4 == 0 and position[1] % 4 == 0:
                cv2.putText(image, f"{position[0]},{position[1]}",
                           (center[0]-20, center[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw walls
    for wall in detector.walls:
        (x1, y1), (x2, y2) = wall
        if y1 < len(detector.grid_cells) and x1 < len(detector.grid_cells[0]) and \
           y2 < len(detector.grid_cells) and x2 < len(detector.grid_cells[0]):
            p1 = detector.grid_cells[y1][x1]["center"]
            p2 = detector.grid_cells[y2][x2]["center"]
            cv2.line(image, p1, p2, (0, 0, 255), 2)

    # Draw robots
    robot_colors = {
        RobotColor.RED: (0, 0, 255),  # BGR
        RobotColor.GREEN: (0, 255, 0),
        RobotColor.BLUE: (255, 0, 0),
        RobotColor.YELLOW: (0, 255, 255),
        RobotColor.BLACK: (0, 0, 0)
    }
    for color, pos in detector.robots.items():
        x, y = pos
        if y < len(detector.grid_cells) and x < len(detector.grid_cells[0]):
            center = detector.grid_cells[y][x]["center"]
            cv2.circle(image, center, 15, robot_colors.get(color, (150, 150, 150)), -1)

            # Add letter on robot
            robot_letter = color.name[0] if hasattr(color, 'name') else '?'
            cv2.putText(image, robot_letter, (center[0]-5, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw targets
    for color, pos in detector.targets.items():
        x, y = pos
        if y < len(detector.grid_cells) and x < len(detector.grid_cells[0]):
            center = detector.grid_cells[y][x]["center"]
            # Draw target as a diamond
            pts = np.array([
                [center[0], center[1]-15],
                [center[0]+15, center[1]],
                [center[0], center[1]+15],
                [center[0]-15, center[1]]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Highlight active target
            if color == detector.active_target:
                cv2.polylines(image, [pts], True, (255, 255, 255), 3)

            cv2.polylines(image, [pts], True, robot_colors.get(color, (150, 150, 150)), 2)

            # Add letter for target
            target_letter = color.name[0].lower() if hasattr(color, 'name') else '?'
            cv2.putText(image, target_letter, (center[0]-5, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_colors.get(color, (150, 150, 150)), 2)

    # Add title
    title = f"Board: {detector.board_size[0]}x{detector.board_size[1]} | "
    title += f"Robots: {len(detector.robots)} | Targets: {len(detector.targets)}"
    cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return image

def main():
    """
    Main function to parse arguments and process the image.
    """
    parser = argparse.ArgumentParser(description="Ricochet Robots Solver")
    parser.add_argument("input_image", help="Path to the input image of the Ricochet Robots board")
    parser.add_argument("--output", "-o", help="Path to save the output image", default="solution.jpg")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output and images")
    parser.add_argument("--config", "-c", help="Path to manual configuration file", default=None)

    args = parser.parse_args()

    start_time = time.time()
    success = process_image(args.input_image, args.output, args.debug, args.config)
    elapsed_time = time.time() - start_time

    if success:
        print(f"Processing completed in {elapsed_time:.2f} seconds")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()