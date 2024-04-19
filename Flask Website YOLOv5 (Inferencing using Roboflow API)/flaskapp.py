from flask import Flask, render_template
from threading import Thread
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

app = Flask(__name__)

# Initialize inference pipeline
pipeline = None

def start_webcam_inference():
    global pipeline
    # Initialize pipeline object
    pipeline = InferencePipeline.init(
        model_id="small-aerial-object-detection-fwbd7/3",  # Roboflow model to use
        video_reference=0,  # Path to video, device id (int, usually 0 for built-in webcams), or RTSP stream URL
        on_prediction=render_boxes,  # Function to run after each prediction
    )
    # Start the pipeline
    pipeline.start()
    # Wait for the pipeline to finish
    pipeline.join()

@app.route('/')
def index():
    return render_template('indexproject.html')


@app.route('/home')
def home():
    return render_template('indexproject.html')


@app.route('/webcam')
def webcam():
    # Start the webcam inference in a separate thread
    Thread(target=start_webcam_inference).start()
    return render_template('webcamPage.html')

if __name__ == '__main__':
    app.run(debug=True)