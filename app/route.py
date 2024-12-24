from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
from detection.Task_Region_Counting import main

app = Flask(__name__)

@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Get paths from request at the start of the function
    video_path = request.form.get('video_path')
    model_path = request.form.get('model_path')
    output_path = request.form.get('output_path')

    # Validate inputs
    if not all([video_path, model_path]):
        return jsonify({
            'error': 'Missing required parameters. Please provide video_path and model_path'
        }), 400

    def generate():
        try:
            for frame in main(video_path, model_path,output_path):
                if frame is not None and isinstance(frame, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Error in video processing: {str(e)}")
            # Yield an error frame or message
            yield b''

    # Return the Response object directly
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/', methods=['GET'])
def index():
    return Response("hello coders")

