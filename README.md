# Ricochet Robots Solver

A computer vision-based tool to solve Ricochet Robots puzzles from a photo.

## Overview

Ricochet Robots is a puzzle board game where robots must be moved to selected locations in as few moves as possible. The robots move in straight lines and only stop when they hit an obstacle (wall or another robot).

This tool analyzes a photo of a Ricochet Robots board, detects the board layout, robots, and targets, then computes and visualizes the optimal solution path.

## Features

- Board detection from photos
- Robot and target detection
- Algorithm to find the optimal solution
- Solution visualization with colored paths and arrows
- Command-line interface
- Web application interface
- Manual board configuration for improved accuracy

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/ricochet-robots-solver.git
cd ricochet-robots-solver
```

2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### Command-line Interface

Process a photo of a Ricochet Robots board:

```
python -m src.main path/to/your/image.jpg --output solution.jpg
```

For debugging image processing:

```
python -m src.main path/to/your/image.jpg --output solution.jpg --debug
```

Use a manual configuration file:

```
python -m src.main path/to/your/image.jpg --output solution.jpg --config board_config.json
```

### Board Verification and Manual Configuration

For more accurate results, you can use the board verification utility:

```
python board_extraction_test.py path/to/your/image.jpg
```

This will create:
- A text representation of the detected board in `board_text.txt`
- A visual verification image in `board_verification.jpg`
- A board configuration file in `board_config.json`

If the automatic detection isn't accurate, you can:
1. Edit the generated `board_config.json` file
2. Or create your own configuration file (see `manual_board_config.json` for a template)
3. Rerun the solver with your manual configuration:

```
python -m src.main path/to/your/image.jpg --config your_config.json
```

For detailed instructions on creating manual configurations, see `MANUAL_CONFIG_INSTRUCTIONS.md`.

### Web Interface

Run the web application:

```
python -m src.webapp
```

Then open a web browser and navigate to `http://localhost:5000`.

## How It Works

1. **Image Processing**: The tool analyzes the input image to detect the board grid, walls, robots, and targets.
2. **Path Finding**: Using a breadth-first search algorithm, it finds the shortest solution path.
3. **Visualization**: The solution is visualized on the original image, showing the path each robot should take.

## Troubleshooting

If you're experiencing issues with the application, try these steps:

1. **Image processing problems**: Run with the `--debug` flag to generate diagnostic images that show the detected grid
   ```
   python -m src.main path/to/your/image.jpg --debug
   ```

2. **Incorrect detection**: Use the manual configuration feature:
   ```
   python board_extraction_test.py path/to/your/image.jpg
   # Edit board_config.json with the correct configuration
   python -m src.main path/to/your/image.jpg --config board_config.json
   ```

3. **No solution found**: Make sure your image is clear and well-lit. If automatic detection fails, try manual configuration.

4. **Web application errors**: Check the console output for error messages. The application logs detailed error information.

5. **Python dependencies**: Make sure all dependencies are installed correctly
   ```
   pip install -r requirements.txt
   ```

## Limitations

- The current image processing algorithm works best with clear, well-lit images
- Board detection is simplified and may require manual configuration for complex boards
- The solver has a maximum move limit to prevent excessive computation

## Future Improvements

- Improved board detection using more advanced computer vision techniques
- Better robot and target recognition
- Support for different board layouts
- Mobile app version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
