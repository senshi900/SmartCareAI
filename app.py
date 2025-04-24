from flask import Flask, request, redirect, render_template, send_from_directory, jsonify, url_for
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil
import pandas as pd
from analyzer import analyze
import ffmpeg
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'mp4'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_name(base_name, ext):
    return f"{Path(base_name).stem}_{uuid.uuid4().hex[:6]}{ext}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        unique_name = generate_unique_name(file.filename, ".mp4")
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        # Run YOLO tracking
        results = model.track(source=filepath, persist=True, tracker="bytetrack.yaml", save=True)

        # Build tracking dataframe
        all_rows = []
        for frame_number, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.xyxy is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

            for i in range(len(xyxy)):
                all_rows.append({
                    "Frame": frame_number,
                    "X1": xyxy[i][0],
                    "Y1": xyxy[i][1],
                    "X2": xyxy[i][2],
                    "Y2": xyxy[i][3],
                    "Player ID": ids[i]
                })

        # Save tracking data with unique name
        csv_name = generate_unique_name("tracking_output", ".csv")
        tracking_csv = os.path.join(RESULT_FOLDER, csv_name)
        tracking_df = pd.DataFrame(all_rows)
        tracking_df.to_csv(tracking_csv, index=False)

        # Analyze
        charts = analyze(tracking_df, RESULT_FOLDER)

        # Convert video result to mp4
        save_dir = Path(results[0].save_dir)
        result_files = list(save_dir.glob("*.avi"))
        if not result_files:
            return "Processed video not found"

        result_path = result_files[0]
        converted_name = generate_unique_name("result", ".mp4")
        converted_path = os.path.join(RESULT_FOLDER, converted_name)

        ffmpeg.input(str(result_path)).output(converted_path, vcodec='libx264', acodec='aac').run(overwrite_output=True)
        os.remove(result_path)

        chart_files = [Path(p).name for p in charts]

        return jsonify({
            "video_url": url_for('result_file', filename=converted_name, _external=True),
            "charts": [url_for('result_file', filename=chart, _external=True) for chart in chart_files]
        })

    return "Invalid file"

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
