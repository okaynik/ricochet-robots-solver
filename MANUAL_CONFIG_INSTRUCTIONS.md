# Manual Board Configuration Instructions

This document provides instructions for manually configuring a Ricochet Robots board when automatic detection isn't accurate enough.

## Board Verification Process

1. First, run the board extraction utility to generate a text representation and debug images:
   ```
   python board_extraction_test.py path/to/your/image.jpg
   ```

2. Examine the output:
   - Text representation in `board_text.txt`
   - Visual representation in `board_verification.jpg`
   - Debug images in the `debug` folder

3. If the automatic detection is not accurate, you can create a manual configuration:
   - Edit `manual_board_config.json` (see format below)
   - Or create a new JSON file with your configuration

4. Run the extraction tool again with your manual configuration:
   ```
   python board_extraction_test.py path/to/your/image.jpg --load-config your_config.json
   ```

## Manual Configuration Format

The configuration file is a JSON file with the following structure:

```json
{
  "board_size": [16, 16],
  "robots": {
    "RED": [3, 7],
    "GREEN": [10, 2],
    "BLUE": [12, 9],
    "YELLOW": [5, 12],
    "BLACK": [8, 4]
  },
  "targets": {
    "RED": [7, 6],
    "GREEN": [14, 3],
    "BLUE": [2, 11],
    "YELLOW": [9, 8]
  },
  "active_target": "RED",
  "walls": [
    [[0, 0], [1, 0]], [[1, 0], [2, 0]],
    [[0, 0], [0, 1]], [[0, 1], [0, 2]],
    [[15, 0], [15, 1]], [[15, 1], [15, 2]],
    [[0, 15], [1, 15]], [[1, 15], [2, 15]],

    [[3, 2], [4, 2]],
    [[7, 5], [7, 6]],
    [[10, 8], [10, 9]],
    [[12, 11], [13, 11]]
  ]
}
```

### Configuration Details

1. `board_size`: Always [16, 16] for standard Ricochet Robots.

2. `robots`: A dictionary mapping robot colors to their positions [x, y] on the board.
   - Valid colors: "RED", "GREEN", "BLUE", "YELLOW", "BLACK"
   - Positions are zero-indexed: [0,0] is the top-left corner, [15,15] is the bottom-right

3. `targets`: A dictionary mapping colors to target positions.
   - Should match the symbols on the board
   - Only include targets that actually exist on the board

4. `active_target`: The current target the robots need to reach.
   - Should be one of the colors in the `targets` dictionary

5. `walls`: A list of wall segments, each defined by its two endpoints.
   - Each wall segment connects two adjacent cells
   - Format: [[x1, y1], [x2, y2]]
   - Include all outer border walls and inner walls

### Wall Definitions

Walls are defined as segments between two adjacent cell coordinates:

- Horizontal wall between (3,5) and (4,5): `[[3, 5], [4, 5]]`
- Vertical wall between (7,8) and (7,9): `[[7, 8], [7, 9]]`

The standard board has walls around the entire perimeter, which should be included.

## Tips for Accurate Configuration

1. Always verify the robot positions by looking at the image
2. Double-check wall placements, as they're critical for correct solutions
3. Make sure the active target matches the one shown in the game
4. The coordinate system is zero-indexed [0,0] to [15,15]
5. After editing the config, run the extraction tool again with `--load-config` to verify