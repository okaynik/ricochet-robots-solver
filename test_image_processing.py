import os
import argparse
import cv2
import numpy as np
from src.image_processing.board_detector import BoardDetector
from src.visualization.path_visualizer import PathVisualizer

def test_processing(image_path, output_path):
    """Test image processing and grid detection"""
    print(f"Testing image processing with: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return False

    try:
        # Load and process image
        board_detector = BoardDetector(image_path)
        success = board_detector.process()

        if not success:
            print("Failed to process board")
            return False

        # Get results
        board_state = board_detector.get_board_state()
        print(f"Board size: {board_state['board_size']}")
        print(f"Detected robots: {len(board_state['robots'])}")
        print(f"Detected walls: {len(board_state['walls'])}")

        # Visualize grid
        grid_image = board_detector.original_image.copy()
        for row in board_detector.grid_cells:
            for cell in row:
                center = cell["center"]
                position = cell["position"]
                cv2.circle(grid_image, center, 3, (0, 255, 0), -1)
                # Add position text for a subset of cells to avoid clutter
                if position[0] % 4 == 0 and position[1] % 4 == 0:
                    cv2.putText(grid_image, f"{position[0]},{position[1]}",
                               (center[0]-20, center[1]+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Save grid visualization
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cv2.imwrite(output_path, grid_image)
        print(f"Grid visualization saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ricochet Robots image processing")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Path for output visualization", default="grid_detection.jpg")

    args = parser.parse_args()
    test_processing(args.input_image, args.output)