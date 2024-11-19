import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No interactive backend
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path, transform="original"):
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")
        
        height, width = gray_image.shape
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]

        if transform == "horizontal_flip":
            gray_image = cv2.flip(gray_image, 1)
        elif transform == "vertical_flip":
            gray_image = cv2.flip(gray_image, 0)
        elif transform == "brightness":
            gray_image = np.clip(gray_image * np.random.uniform(1.5, 2.0), 0, 255)

        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(gray_image, cmap='gray')
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if transform == "horizontal_flip":
                x = width - x
            plt.plot(x, y, 'rx')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files.get('file')
        transform = request.form.get('transform', 'original')
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file or no file provided'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result_image = analyze_face(filepath, transform=transform)
        return jsonify({'success': True, 'image': result_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

