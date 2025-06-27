from flask import Flask, Response, render_template
import cv2
from age_gender_detection import process_frame  # Adjust based on your function names

app = Flask(__name__)

def generate_video():
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Process the frame to detect age and gender
        frame = process_frame(frame)

        # Encode the frame in JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame in a format that Flask can render
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')  # Make sure you have this HTML file

if __name__ == '__main__':
    app.run(debug=True)
