from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames(neck_image_path):
    # Load the image to overlay
    neck_image = cv2.imread(neck_image_path, cv2.IMREAD_UNCHANGED)

    # Start video capture from the camera
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        try:
            # Process each detected face
            for (x, y, w, h) in faces:
                # Estimate the neck position and dimensions
                neck_top = y + h
                neck_height = int(h * 0.8)
                neck_width = w

                # Resize the neck image to fit the neck area
                resized_neck_image = cv2.resize(neck_image, (neck_width, neck_height))

                # Create a mask and its inverse mask from the alpha channel
                alpha_s = resized_neck_image[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                # Blend the overlay image with the frame
                for c in range(0, 3):
                    frame[neck_top:neck_top + neck_height, x:x + neck_width, c] = (alpha_s * resized_neck_image[:, :, c] +
                                                                                     alpha_l * frame[neck_top:neck_top + neck_height, x:x + neck_width, c])
        except Exception:
            pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed/<image>')
def video_feed(image):
    # Replace 'path/to/neck_image.png' with the actual path to your overlay image
    # image = 'image3.png'
    return Response(gen_frames(image), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
