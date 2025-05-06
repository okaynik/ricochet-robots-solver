import heapq
from typing import Dict, List, Tuple, Set, Optional
from collections import deque
import networkx as nx
from src.image_processing.board_detector import RobotColor, Direction

class PathFinder:
    """
    Finds the optimal solution path for a Ricochet Robots puzzle.
    """

    def __init__(self, board_state: Dict):
        """
        Initialize the path finder with the board state.

        Args:
            board_state: Dictionary containing board information
        """
        self.board_size = board_state["board_size"]
        self.walls = board_state["walls"]
        self.robots = board_state["robots"]
        self.targets = board_state["targets"]
        self.active_target = board_state["active_target"]

        # Convert walls to more usable format
        self.wall_lookup = self._process_walls()

    def _process_walls(self) -> Dict:
        """
        Convert walls list to a lookup dictionary for efficient access.

        Returns:
            Dictionary mapping cell positions to sets of blocked directions
        """
        wall_dict = {}

        for wall in self.walls:
            (x1, y1), (x2, y2) = wall

            # Horizontal wall
            if y1 == y2:
                # Wall is above cells in row y1
                if (x1, y1) not in wall_dict:
                    wall_dict[(x1, y1)] = set()
                wall_dict[(x1, y1)].add(Direction.DOWN)

                # Wall is below cells in row y1-1
                if (x1, y1-1) not in wall_dict:
                    wall_dict[(x1, y1-1)] = set()
                wall_dict[(x1, y1-1)].add(Direction.UP)

            # Vertical wall
            elif x1 == x2:
                # Wall is to the right of cells in column x1
                if (x1, y1) not in wall_dict:
                    wall_dict[(x1, y1)] = set()
                wall_dict[(x1, y1)].add(Direction.RIGHT)

                # Wall is to the left of cells in column x1+1
                if (x1+1, y1) not in wall_dict:
                    wall_dict[(x1+1, y1)] = set()
                wall_dict[(x1+1, y1)].add(Direction.LEFT)

        return wall_dict

    def _is_blocked(self, pos: Tuple[int, int], direction: Direction, robot_positions: Dict[RobotColor, Tuple[int, int]]) -> bool:
        """
        Check if movement in a given direction is blocked by a wall or another robot.

        Args:
            pos: Current position (x, y)
            direction: Direction of movement
            robot_positions: Dictionary mapping robot colors to positions

        Returns:
            True if movement is blocked, False otherwise
        """
        # Check wall
        if pos in self.wall_lookup and direction in self.wall_lookup[pos]:
            return True

        # Check board boundaries
        x, y = pos
        if (direction == Direction.UP and y == 0) or \
           (direction == Direction.RIGHT and x == self.board_size[0] - 1) or \
           (direction == Direction.DOWN and y == self.board_size[1] - 1) or \
           (direction == Direction.LEFT and x == 0):
            return True

        # Check other robots
        next_pos = self._get_next_position(pos, direction)
        for robot_pos in robot_positions.values():
            if next_pos == robot_pos:
                return True

        return False

    def _get_next_position(self, pos: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        """
        Get the next position when moving in a given direction.

        Args:
            pos: Current position (x, y)
            direction: Direction of movement

        Returns:
            Next position (x, y)
        """
        x, y = pos
        if direction == Direction.UP:
            return (x, y - 1)
        elif direction == Direction.RIGHT:
            return (x + 1, y)
        elif direction == Direction.DOWN:
            return (x, y + 1)
        elif direction == Direction.LEFT:
            return (x - 1, y)
        else:
            return pos

    def _move_robot(self, robot_pos: Tuple[int, int], direction: Direction, robot_positions: Dict[RobotColor, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Move a robot in a given direction until it hits a wall or another robot.

        Args:
            robot_pos: Current position of the robot
            direction: Direction to move
            robot_positions: Dictionary mapping robot colors to positions

        Returns:
            New position of the robot
        """
        current_pos = robot_pos

        while not self._is_blocked(current_pos, direction, robot_positions):
            current_pos = self._get_next_position(current_pos, direction)

        return current_pos

    def find_solution(self) -> Optional[List[Tuple[RobotColor, Direction]]]:
        """
        Find the shortest solution for the puzzle.

        Returns:
            List of (robot, direction) moves for the solution, or None if no solution found
        """
        # Create initial state
        initial_state = frozenset(self.robots.items())
        target_color = self.active_target
        target_pos = self.targets[target_color]

        # Create a queue for BFS
        queue = deque([(initial_state, [])])

        # Keep track of visited states
        visited = set([initial_state])

        # Maximum number of moves to try
        max_moves = 20

        while queue:
            state, moves = queue.popleft()

            # Convert state back to dictionary for easier handling
            robot_positions = dict(state)

            # Check if we've reached the target
            if robot_positions[target_color] == target_pos:
                return moves

            # Check if we've reached the maximum number of moves
            if len(moves) >= max_moves:
                continue

            # Generate all possible moves
            for robot_color in robot_positions:
                robot_pos = robot_positions[robot_color]

                for direction in Direction:
                    # Skip moves that don't change position
                    new_pos = self._move_robot(robot_pos, direction, robot_positions)
                    if new_pos == robot_pos:
                        continue

                    # Create new state
                    new_robot_positions = robot_positions.copy()
                    new_robot_positions[robot_color] = new_pos
                    new_state = frozenset(new_robot_positions.items())

                    # Skip if we've already visited this state
                    if new_state in visited:
                        continue

                    # Add to queue and visited set
                    new_moves = moves + [(robot_color, direction)]
                    queue.append((new_state, new_moves))
                    visited.add(new_state)

        # No solution found
        return None

    def get_path_coordinates(self, moves: List[Tuple[RobotColor, Direction]]) -> List[Tuple[RobotColor, List[Tuple[int, int]]]]:
        """
        Convert a solution into a list of coordinates for each robot's path.

        Args:
            moves: List of (robot, direction) moves

        Returns:
            List of (robot, path) tuples where path is a list of coordinates
        """
        paths = {robot: [pos] for robot, pos in self.robots.items()}
        robot_positions = self.robots.copy()

        for robot, direction in moves:
            current_pos = robot_positions[robot]
            new_pos = self._move_robot(current_pos, direction, robot_positions)

            # Generate intermediate positions for the path
            intermediate_positions = []
            pos = current_pos

            while pos != new_pos:
                pos = self._get_next_position(pos, direction)
                intermediate_positions.append(pos)

                # Stop if we've reached the destination
                if pos == new_pos:
                    break

            # Update paths and robot positions
            paths[robot].extend(intermediate_positions)
            robot_positions[robot] = new_pos

        return [(robot, path) for robot, path in paths.items()]