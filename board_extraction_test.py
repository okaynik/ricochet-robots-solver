import os
import argparse
import cv2
import numpy as np
import json
from src.image_processing.board_detector import BoardDetector, RobotColor, Direction

def detect_board_elements(image_path):
    """
    Detect board elements and return structured data
    """
    print(f"Analyzing image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return None

    try:
        # Load and process image
        board_detector = BoardDetector(image_path)
        success = board_detector.process()

        if not success:
            print("Failed to process board")
            return None

        return {
            "board_size": board_detector.board_size,
            "grid_cells": board_detector.grid_cells,
            "walls": board_detector.walls,
            "robots": board_detector.robots,
            "targets": board_detector.targets,
            "active_target": board_detector.active_target,
            "original_image": board_detector.original_image,
            "debug_images": board_detector.debug_images
        }

    except Exception as e:
        print(f"Error detecting board elements: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_text_representation(board_data):
    """
    Create a text representation of the board
    """
    if not board_data:
        return "Failed to create text representation: No board data available"

    board_size = board_data["board_size"]
    walls = board_data["walls"]
    robots = board_data["robots"]
    targets = board_data["targets"]
    active_target = board_data["active_target"]

    # Create the horizontal walls representation
    horizontal_walls = set()
    vertical_walls = set()

    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if y1 == y2:  # Horizontal wall
            horizontal_walls.add((min(x1, x2), y1, max(x1, x2), y2))
        else:  # Vertical wall
            vertical_walls.add((x1, min(y1, y2), x2, max(y1, y2)))

    # Create a text grid with borders
    # 2x the board size + 1 for borders
    grid_width = board_size[0] * 2 + 1
    grid_height = board_size[1] * 2 + 1

    # Initialize empty grid
    text_grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    # Fill in the grid cells
    for y in range(board_size[1]):
        for x in range(board_size[0]):
            # Each cell's center position in text grid
            center_y = y * 2 + 1
            center_x = x * 2 + 1

            # Place dots for empty cells
            text_grid[center_y][center_x] = '·'

    # Draw outer borders
    for x in range(grid_width):
        text_grid[0][x] = '─' if x % 2 == 1 else '┬' if x > 0 and x < grid_width - 1 else '┌' if x == 0 else '┐'
        text_grid[grid_height-1][x] = '─' if x % 2 == 1 else '┴' if x > 0 and x < grid_width - 1 else '└' if x == 0 else '┘'

    for y in range(grid_height):
        if y % 2 == 1:
            text_grid[y][0] = '│'
            text_grid[y][grid_width-1] = '│'
        else:
            text_grid[y][0] = '├' if y > 0 and y < grid_height - 1 else text_grid[y][0]
            text_grid[y][grid_width-1] = '┤' if y > 0 and y < grid_height - 1 else text_grid[y][grid_width-1]

    # Draw inner grid lines
    for y in range(1, grid_height-1):
        for x in range(1, grid_width-1):
            if x % 2 == 0 and y % 2 == 0:
                text_grid[y][x] = '┼'  # Grid intersection
            elif x % 2 == 0:
                text_grid[y][x] = '│'  # Vertical grid line
            elif y % 2 == 0:
                text_grid[y][x] = '─'  # Horizontal grid line

    # Add horizontal walls
    for x1, y1, x2, y2 in horizontal_walls:
        # Wall position in text grid
        grid_y = y1 * 2
        for grid_x in range(x1 * 2 + 1, (x2 * 2) + 1):
            if grid_y >= 0 and grid_y < grid_height and grid_x >= 0 and grid_x < grid_width:
                text_grid[grid_y][grid_x] = '═' if grid_x % 2 == 1 else '╪'

    # Add vertical walls
    for x1, y1, x2, y2 in vertical_walls:
        # Wall position in text grid
        grid_x = x1 * 2
        for grid_y in range(y1 * 2 + 1, (y2 * 2) + 1):
            if grid_y >= 0 and grid_y < grid_height and grid_x >= 0 and grid_x < grid_width:
                text_grid[grid_y][grid_x] = '║' if grid_y % 2 == 1 else '╫'

    # Robot symbol mapping
    robot_symbols = {
        RobotColor.RED: 'R',
        RobotColor.GREEN: 'G',
        RobotColor.BLUE: 'B',
        RobotColor.YELLOW: 'Y',
        RobotColor.BLACK: 'K'
    }

    # Target symbol mapping
    target_symbols = {
        RobotColor.RED: 'r',
        RobotColor.GREEN: 'g',
        RobotColor.BLUE: 'b',
        RobotColor.YELLOW: 'y',
        RobotColor.BLACK: 'k'
    }

    # Add robots to the grid
    for color, pos in robots.items():
        x, y = pos
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            text_grid[y * 2 + 1][x * 2 + 1] = robot_symbols.get(color, '?')

    # Add targets to the grid
    for color, pos in targets.items():
        x, y = pos
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            # Check if the cell has a robot already
            if text_grid[y * 2 + 1][x * 2 + 1] == '·':
                text_grid[y * 2 + 1][x * 2 + 1] = target_symbols.get(color, '?')
            else:
                # If the cell has a robot, mark the target with brackets
                text_grid[y * 2 + 1][x * 2 + 1] = f'({text_grid[y * 2 + 1][x * 2 + 1]})'

    # Highlight active target if known
    if active_target and active_target in targets:
        x, y = targets[active_target]
        if 0 <= x < board_size[0] and 0 <= y < board_size[1]:
            symbol = text_grid[y * 2 + 1][x * 2 + 1]
            if len(symbol) == 1:  # Not already marked with brackets
                text_grid[y * 2 + 1][x * 2 + 1] = f'*{symbol}*'

    # Convert grid to string
    text_representation = '\n'.join([''.join(row) for row in text_grid])

    # Add legend
    legend = "\nLEGEND:\n"
    legend += "R, G, B, Y, K = Red, Green, Blue, Yellow, Black robots\n"
    legend += "r, g, b, y, k = Red, Green, Blue, Yellow, Black targets\n"
    legend += "· = Empty cell\n"
    legend += "║, ═ = Walls\n"
    legend += "*x* = Active target\n"

    # Add detected elements summary
    summary = "\nDETECTED ELEMENTS:\n"
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

    return text_representation + legend + summary

def create_visual_debug_image(board_data, output_path):
    """
    Create a visual debug image showing detected elements
    """
    if not board_data:
        print("Failed to create visual debug: No board data available")
        return False

    image = board_data["original_image"].copy()
    grid_cells = board_data["grid_cells"]
    walls = board_data["walls"]
    robots = board_data["robots"]
    targets = board_data["targets"]
    active_target = board_data["active_target"]

    # Draw grid
    for row in grid_cells:
        for cell in row:
            center = cell["center"]
            position = cell["position"]
            cv2.circle(image, center, 2, (100, 100, 100), -1)

            # Draw position coordinates on some cells
            if position[0] % 4 == 0 and position[1] % 4 == 0:
                cv2.putText(image, f"{position[0]},{position[1]}",
                           (center[0]-15, center[1]+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 200), 1)

    # Draw walls
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        # Convert grid coordinates to pixel coordinates
        if y1 < len(grid_cells) and x1 < len(grid_cells[0]):
            p1_center = grid_cells[y1][x1]["center"] if y1 < len(grid_cells) and x1 < len(grid_cells[0]) else (0, 0)
            p2_center = grid_cells[y2][x2]["center"] if y2 < len(grid_cells) and x2 < len(grid_cells[0]) else (0, 0)
            cv2.line(image, p1_center, p2_center, (0, 0, 255), 2)

    # Draw robots
    robot_colors = {
        RobotColor.RED: (0, 0, 255),  # BGR format
        RobotColor.GREEN: (0, 255, 0),
        RobotColor.BLUE: (255, 0, 0),
        RobotColor.YELLOW: (0, 255, 255),
        RobotColor.BLACK: (0, 0, 0)
    }

    for color, pos in robots.items():
        x, y = pos
        if y < len(grid_cells) and x < len(grid_cells[0]):
            center = grid_cells[y][x]["center"]
            cv2.circle(image, center, 15, robot_colors.get(color, (150, 150, 150)), -1)

            # Add letter on top of robot
            robot_letter = color.name[0] if hasattr(color, 'name') else '?'
            cv2.putText(image, robot_letter, (center[0]-5, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw targets
    for color, pos in targets.items():
        x, y = pos
        if y < len(grid_cells) and x < len(grid_cells[0]):
            center = grid_cells[y][x]["center"]
            # Draw target as a diamond
            pts = np.array([
                [center[0], center[1]-15],
                [center[0]+15, center[1]],
                [center[0], center[1]+15],
                [center[0]-15, center[1]]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Highlight the active target
            if color == active_target:
                cv2.polylines(image, [pts], True, (255, 255, 255), 3)

            cv2.polylines(image, [pts], True, robot_colors.get(color, (150, 150, 150)), 2)

            # Add letter for target
            target_letter = color.name[0].lower() if hasattr(color, 'name') else '?'
            cv2.putText(image, target_letter, (center[0]-5, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_colors.get(color, (150, 150, 150)), 2)

    # Add title with detection summary
    title = f"Board: {board_data['board_size'][0]}x{board_data['board_size'][1]} | "
    title += f"Robots: {len(robots)} | Targets: {len(targets)} | Walls: {len(walls)}"
    cv2.putText(image, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save image
    cv2.imwrite(output_path, image)
    print(f"Visual debug image saved to: {output_path}")

    # Also save debug images if available
    if "debug_images" in board_data and board_data["debug_images"]:
        debug_dir = os.path.join(os.path.dirname(output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        for name, debug_img in board_data["debug_images"].items():
            debug_path = os.path.join(debug_dir, f"{name}_{os.path.basename(output_path)}")
            cv2.imwrite(debug_path, debug_img)
            print(f"Debug image '{name}' saved to: {debug_path}")

    return True

def save_board_config(board_data, output_path):
    """
    Save board configuration to a JSON file for later use
    """
    if not board_data:
        print("Failed to save config: No board data available")
        return False

    # Convert the board data to a serializable format
    serializable_data = {
        "board_size": board_data["board_size"],
        "walls": board_data["walls"],
        "robots": {k.name if hasattr(k, 'name') else str(k): v for k, v in board_data["robots"].items()},
        "targets": {k.name if hasattr(k, 'name') else str(k): v for k, v in board_data["targets"].items()},
        "active_target": board_data["active_target"].name if hasattr(board_data["active_target"], 'name') else str(board_data["active_target"]) if board_data["active_target"] else None
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Board configuration saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving board configuration: {str(e)}")
        return False

def load_board_config(config_path):
    """
    Load board configuration from a JSON file
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
                robots[color] = v
            except (KeyError, ValueError):
                robots[k] = v

        targets = {}
        for k, v in data["targets"].items():
            try:
                color = RobotColor[k] if k in [e.name for e in RobotColor] else k
                targets[color] = v
            except (KeyError, ValueError):
                targets[k] = v

        active_target = None
        if data["active_target"]:
            try:
                active_target = RobotColor[data["active_target"]] if data["active_target"] in [e.name for e in RobotColor] else data["active_target"]
            except (KeyError, ValueError):
                active_target = data["active_target"]

        return {
            "board_size": tuple(data["board_size"]),
            "walls": [tuple(tuple(p) for p in wall) for wall in data["walls"]],
            "robots": robots,
            "targets": targets,
            "active_target": active_target
        }
    except Exception as e:
        print(f"Error loading board configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def update_board_with_manual_config(board_data, config_data):
    """
    Update board data with manually specified configuration data
    """
    if not board_data or not config_data:
        return board_data

    # Keep the original image and grid_cells
    original_image = board_data.get("original_image")
    grid_cells = board_data.get("grid_cells")
    debug_images = board_data.get("debug_images", {})

    # Update with the config data
    updated_data = {
        "board_size": config_data.get("board_size", board_data.get("board_size")),
        "walls": config_data.get("walls", board_data.get("walls")),
        "robots": config_data.get("robots", board_data.get("robots")),
        "targets": config_data.get("targets", board_data.get("targets")),
        "active_target": config_data.get("active_target", board_data.get("active_target")),
        "original_image": original_image,
        "grid_cells": grid_cells,
        "debug_images": debug_images
    }

    return updated_data

def main():
    parser = argparse.ArgumentParser(description="Extract and verify Ricochet Robots board elements")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Path for output visualization", default="board_verification.jpg")
    parser.add_argument("--text", "-t", help="Path for text output file", default="board_text.txt")
    parser.add_argument("--config", "-c", help="Path for board configuration file (JSON)", default="board_config.json")
    parser.add_argument("--load-config", "-l", help="Load configuration from file instead of detecting", default=None)

    args = parser.parse_args()

    # Detect or load board elements
    if args.load_config:
        print(f"Loading board configuration from: {args.load_config}")
        config_data = load_board_config(args.load_config)
        if not config_data:
            print("Failed to load configuration")
            return

        # We still need the original image and grid_cells for visualization
        # So we'll detect the board and then update with the manual config
        board_data = detect_board_elements(args.input_image)
        if board_data:
            board_data = update_board_with_manual_config(board_data, config_data)
    else:
        board_data = detect_board_elements(args.input_image)

    if board_data:
        # Create and save text representation
        text_representation = create_text_representation(board_data)

        # Print to console
        print("\nBOARD TEXT REPRESENTATION:")
        print(text_representation)

        # Save to file
        with open(args.text, "w") as f:
            f.write(text_representation)
        print(f"Text representation saved to: {args.text}")

        # Create and save visual debug image
        create_visual_debug_image(board_data, args.output)

        # Save board configuration
        save_board_config(board_data, args.config)
    else:
        print("Failed to extract board elements")

if __name__ == "__main__":
    main()