import os
import tempfile
import uuid
import traceback
from flask import Flask, request, render_template, send_from_directory, url_for, flash
from werkzeug.utils import secure_filename
from src.main import process_image

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = os.urandom(24)  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'results')
DEBUG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'debug')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['DEBUG_FOLDER'] = DEBUG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/debug/<filename>')
def debug_file(filename):
    """Serve debug files."""
    return send_from_directory(app.config['DEBUG_FOLDER'], filename)

@app.route('/solve', methods=['POST'])
def solve():
    """Process the uploaded image and display the solution."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected')

    file = request.files['file']

    # Check if the user submitted an empty form
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    # Check if the file is valid
    if not file or not allowed_file(file.filename):
        return render_template('index.html',
                               error='Invalid file format. Please upload a JPG, JPEG or PNG image')

    try:
        # Generate unique filename
        filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename).rsplit('.', 1)[1].lower()
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_filename = 'solution_' + filename
        output_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        debug_filename = 'debug_' + filename
        debug_path = os.path.join(app.config['DEBUG_FOLDER'], debug_filename)

        # Save the file
        file.save(input_path)

        # Process the image
        success = process_image(input_path, output_path, debug=True)

        if success:
            return render_template('result.html',
                                input_image=url_for('uploaded_file', filename=filename),
                                result_image=url_for('result_file', filename=result_filename),
                                debug_image=url_for('debug_file', filename='debug_grid_' + filename))
        else:
            return render_template('index.html',
                                error='Failed to process the image. Please try another image or check server logs.')

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        app.logger.error(error_message)
        app.logger.error(traceback.format_exc())
        return render_template('index.html', error=error_message)

@app.errorhandler(500)
def server_error(e):
    """Handle internal server errors."""
    app.logger.error(f"Server error: {str(e)}")
    return render_template('index.html', error="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True)