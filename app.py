from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from subprocess import Popen, PIPE
import os
import secrets
import logging
import subprocess
from flask import Flask, jsonify
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Specify the absolute path to your YOLOv5 model folder
yolov5_path = os.path.join(os.getcwd(), "yolov5")

# Set up the absolute paths to the YOLOv5 detection script and best.pt file
detection_script = os.path.join(yolov5_path, "detect.py")
best_pt_path = os.path.join(
    yolov5_path, "runs", "train", "exp4", "weights", "best.pt")

# Set up the absolute path to the directory where YOLOv5 will save the results
output_path = os.path.join(yolov5_path, "runs", "detect")

# Set up the upload folder for images
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class UploadForm(FlaskForm):
    file = FileField('Image File', validators=[FileRequired(
    ), FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])


def get_latest_exp_folder(base_path):
    # Specify the path to the directory (e.g., C:\Users\Imraan\OneDrive\Desktop\clear_view\yolov5\runs\detect)
    detect_path = os.path.join(base_path, "runs", "detect")

    # Check if the directory exists
    if not os.path.exists(detect_path):
        print(f"Directory does not exist: {detect_path}")
        return None, None

    # Get a list of all directories in the detect_path
    subfolders = [f.name for f in os.scandir(detect_path) if f.is_dir()]

    # Sort the directories in increasing order
    sorted_folders = sorted(subfolders, key=lambda x: int(
        x[3:]) if x[3:].isdigit() else float('inf'))

    # Retrieve the last folder
    if (len(sorted_folders)) == 0:
        return None
    elif len(sorted_folders) == 1:
        latest_folder = sorted_folders[-1]
    else:
        latest_folder = sorted_folders[-2]

    # Check if the latest folder is found
    if latest_folder:
        # Get the full path of the "input_image" file within the latest folder
        input_image_path = os.path.join(
            detect_path, latest_folder, "input_image.jpg")
        return latest_folder, input_image_path
    else:
        return None, None


def run_yolov5_detection(filepath):
    try:
        # Log the absolute path to the YOLOv5 script and best.pt file
        logging.debug(
            f"YOLOv5 script path: {os.path.abspath(detection_script)}")
        logging.debug(f"best.pt path: {os.path.abspath(best_pt_path)}")

        modified_filepath = filepath.replace(
            r"C:\Users\Imraan\OneDrive\Desktop\clear_view\yolov5\static\uploads",
            r"C:\Users\Imraan\OneDrive\Desktop\clear_view\static\uploads"
        )

        command = f"python {os.path.abspath(detection_script)} --weights {os.path.abspath(best_pt_path)} --img-size 640 --conf 0.4 --source {modified_filepath} --project {os.path.abspath(output_path)}"
        logging.debug(f"YOLOv5 detection command: {command}")

        process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        output, error = process.communicate()

        # Check if the command was successful
        if process.returncode == 0:
            return True, output.decode("utf-8")
        else:
            error_message = error.decode('utf-8')
            logging.error(f"YOLOv5 Error: {error_message}")
            return False, error_message
    except Exception as e:
        logging.error(f"Exception during YOLOv5 detection: {str(e)}")
        return False, str(e)


def run_yolov5_detection_webcam():
    try:
        # Log the absolute path to the YOLOv5 script and best.pt file
        logging.debug(f"YOLOv5 script path: {os.path.abspath(detection_script)}")
        logging.debug(f"best.pt path: {os.path.abspath(best_pt_path)}")

        command = f"python {os.path.abspath(detection_script)} --weights {os.path.abspath(best_pt_path)} --img-size 640  --source 0 --project {os.path.abspath(output_path)}"
        logging.debug(f"YOLOv5 detection command: {command}")

        process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        return process  # Always return the process object
    except Exception as e:
        logging.error(f"Exception during YOLOv5 detection: {str(e)}")
        return None  # Return None in case of an exception



@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    if form.validate_on_submit():
        file = form.file.data
        filename = os.path.join(app.config["UPLOAD_FOLDER"], "input_image.jpg")
        file.save(filename)

        success, results = run_yolov5_detection(filename)

        if success:
            processed_results = results

            # Get the latest 'exp' folder and 'input_image' path
            latest_exp_folder, input_image_path = get_latest_exp_folder(
                yolov5_path)
            print(input_image_path)

            # Update the rendered template with the new paths
            return render_template("index.html", form=form, results=processed_results, image="input_image.jpg", output_image_path=input_image_path)
        else:
            flash(f"Error running YOLOv5 detection: {results}", 'error')
            return redirect(url_for('index'))

    return render_template("index.html", form=form, results=None, image=None)


@app.route('/output_image')
def output_image():
    # Assuming that output_image_path contains the full path to the image
    output_image_path = request.args.get('output_image_path', '')
    return send_from_directory(os.path.dirname(output_image_path), os.path.basename(output_image_path))


process = None

@app.route('/start-yolo-detection-webcam', methods=['POST'])
def start_yolo_detection_webcam():
    global process
    
    process =run_yolov5_detection_webcam()
    # process = subprocess.Popen(["python", "your_detection_script.py"])
    return render_template("index.html", process=process)

@app.route('/stop-yolo-detection-webcam', methods=['POST'])
def stop_yolo_detection_webcam():
    global process
    if process:
        process.kill()
        process = None
    return "Process stopped successfully"

if __name__ == "__main__":
    app.run(debug=True)
