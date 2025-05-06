#!/usr/bin/env python3
"""
Test script for Ricochet Robots solver.
This script uses a sample configuration to test the solver without the need for image processing.
"""

import os
import json
import time
import argparse
from src.image_processing.board_detector import RobotColor
from src.solver.path_finder import PathFinder

# Sample configuration for a simple Ricochet Robots board
SAMPLE_CONFIG = {
    "board_size": (16, 16),
    "robots": {
        RobotColor.RED: (3, 7),
        RobotColor.GREEN: (10, 2),
        RobotColor.BLUE: (12, 9),
        RobotColor.YELLOW: (5, 12)
    },
    "targets": {
        RobotColor.RED: (7, 6),
        RobotColor.GREEN: (14, 3),
        RobotColor.BLUE: (2, 11),
        RobotColor.YELLOW: (9, 8)
    },
    "active_target": RobotColor.RED,
    "walls": [
        # Outer borders - top
        ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)),
        ((4, 0), (5, 0)), ((5, 0), (6, 0)), ((6, 0), (7, 0)), ((7, 0), (8, 0)),
        ((8, 0), (9, 0)), ((9, 0), (10, 0)), ((10, 0), (11, 0)), ((11, 0), (12, 0)),
        ((12, 0), (13, 0)), ((13, 0), (14, 0)), ((14, 0), (15, 0)),

        # Outer borders - right
        ((15, 0), (15, 1)), ((15, 1), (15, 2)), ((15, 2), (15, 3)), ((15, 3), (15, 4)),
        ((15, 4), (15, 5)), ((15, 5), (15, 6)), ((15, 6), (15, 7)), ((15, 7), (15, 8)),
        ((15, 8), (15, 9)), ((15, 9), (15, 10)), ((15, 10), (15, 11)), ((15, 11), (15, 12)),
        ((15, 12), (15, 13)), ((15, 13), (15, 14)), ((15, 14), (15, 15)),

        # Outer borders - bottom
        ((0, 15), (1, 15)), ((1, 15), (2, 15)), ((2, 15), (3, 15)), ((3, 15), (4, 15)),
        ((4, 15), (5, 15)), ((5, 15), (6, 15)), ((6, 15), (7, 15)), ((7, 15), (8, 15)),
        ((8, 15), (9, 15)), ((9, 15), (10, 15)), ((10, 15), (11, 15)), ((11, 15), (12, 15)),
        ((12, 15), (13, 15)), ((13, 15), (14, 15)), ((14, 15), (15, 15)),

        # Outer borders - left
        ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 3), (0, 4)),
        ((0, 4), (0, 5)), ((0, 5), (0, 6)), ((0, 6), (0, 7)), ((0, 7), (0, 8)),
        ((0, 8), (0, 9)), ((0, 9), (0, 10)), ((0, 10), (0, 11)), ((0, 11), (0, 12)),
        ((0, 12), (0, 13)), ((0, 13), (0, 14)), ((0, 14), (0, 15)),

        # Inner walls - just a few examples
        ((3, 2), (4, 2)),
        ((7, 5), (7, 6)),
        ((10, 8), (10, 9)),
        ((12, 11), (13, 11))
    ]
}

def load_config_from_file(config_path):
    """
    Load a board configuration from a JSON file

    Args:
        config_path: Path to the configuration file

    Returns:
        Board configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Convert the JSON data to the expected format
        robots = {}
        for k, v in data.get("robots", {}).items():
            try:
                color = getattr(RobotColor, k) if hasattr(RobotColor, k) else k
                robots[color] = tuple(v)
            except (AttributeError, TypeError):
                robots[k] = tuple(v)

        targets = {}
        for k, v in data.get("targets", {}).items():
            try:
                color = getattr(RobotColor, k) if hasattr(RobotColor, k) else k
                targets[color] = tuple(v)
            except (AttributeError, TypeError):
                targets[k] = tuple(v)

        active_target = None
        if "active_target" in data and data["active_target"]:
            try:
                active_target = getattr(RobotColor, data["active_target"]) if hasattr(RobotColor, data["active_target"]) else data["active_target"]
            except (AttributeError, TypeError):
                active_target = data["active_target"]

        # Convert walls to tuples of tuples
        walls = []
        for wall in data.get("walls", []):
            walls.append(tuple(tuple(p) for p in wall))

        return {
            "board_size": tuple(data.get("board_size", (16, 16))),
            "robots": robots,
            "targets": targets,
            "active_target": active_target,
            "walls": walls
        }
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def solve_board(board_state):
    """
    Solve a Ricochet Robots board

    Args:
        board_state: Board state dictionary

    Returns:
        List of moves, or None if no solution found
    """
    print("Board configuration:")
    print(f"Board size: {board_state['board_size']}")
    print(f"Robots: {len(board_state['robots'])}")
    print(f"Robot positions: {', '.join([f'{k.name if hasattr(k, 'name') else k}:{v}' for k, v in board_state['robots'].items()])}")
    print(f"Targets: {len(board_state['targets'])}")
    print(f"Target positions: {', '.join([f'{k.name if hasattr(k, 'name') else k}:{v}' for k, v in board_state['targets'].items()])}")
    print(f"Active target: {board_state['active_target'].name if hasattr(board_state['active_target'], 'name') else board_state['active_target']}")
    print(f"Walls: {len(board_state['walls'])}")

    # Solve the board
    print("\nFinding solution...")
    start_time = time.time()

    path_finder = PathFinder(board_state)
    solution = path_finder.find_solution()

    elapsed_time = time.time() - start_time

    if solution is None:
        print("No solution found.")
        return None

    # Print the solution
    print(f"Solution found in {elapsed_time:.2f} seconds:")
    print(f"Total moves: {len(solution)}")

    print("\nSolution steps:")
    for i, (robot, direction) in enumerate(solution):
        robot_name = robot.name if hasattr(robot, 'name') else robot
        direction_name = direction.name if hasattr(direction, 'name') else direction
        print(f"Step {i+1}: Move {robot_name} {direction_name}")

    # Get the path coordinates
    paths = path_finder.get_path_coordinates(solution)

    print("\nRobot paths:")
    for robot, path in paths:
        robot_name = robot.name if hasattr(robot, 'name') else robot
        path_str = ' -> '.join([f"({x},{y})" for x, y in path])
        print(f"{robot_name}: {path_str}")

    return solution

def main():
    parser = argparse.ArgumentParser(description="Test Ricochet Robots solver")
    parser.add_argument("--config", "-c", help="Path to configuration file (JSON)", default=None)

    args = parser.parse_args()

    # Use either the provided config file or the sample config
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}...")
        board_state = load_config_from_file(args.config)
        if not board_state:
            print("Failed to load configuration, using sample configuration.")
            board_state = SAMPLE_CONFIG
    else:
        print("Using sample configuration...")
        board_state = SAMPLE_CONFIG

    # Solve the board
    solution = solve_board(board_state)

    if solution:
        print("\nTest successful!")
    else:
        print("\nTest failed: No solution found.")

if __name__ == "__main__":
    main()