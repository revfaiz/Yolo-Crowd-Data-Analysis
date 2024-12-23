from flask import Flask, Response, request, jsonify
import cv2
import asyncio
import numpy as np
from detection.Task_Region_Counting import process_detections_user_marked

app = Flask(__name__)

@app.route('/video_feed', methods=['POST'])
def video_feed():
    def generate():
        video_path = r"D:\Yolo Implementation\gRPC\Sample_Video1.mp4"
        model_path = r"D:\Yolo Implementation\gRPC\yolov8n.pt"

        try:
            for frame in process_detections_user_marked(video_path=video_path, model_path=model_path):
                print("I am in api call hoorah")
                if frame is not None and isinstance(frame, np.ndarray):
                    print("I am in if condition")
                    _, buffer = cv2.imencode('.jpg', frame)
                    print("I am in buffer")
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    print("I am in yield")
        except Exception as e:
            print(f"Fatal error in video processing: {str(e)}")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET'])
def index():
    return Response("hello coders")

