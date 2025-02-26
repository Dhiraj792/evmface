import cv2
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request
import os
import webbrowser
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

app = Flask(__name__)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure the "data" directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Load registered faces and Aadhaar details
try:
    with open("data/names.pkl", "rb") as f:
        names = pickle.load(f)
    with open("data/faces_data.pkl", "rb") as f:
        faces_data = pickle.load(f)
    with open("data/aadhaar_data.pkl", "rb") as f:
        aadhaar_data = pickle.load(f)

    faces_data = np.array(faces_data).reshape(len(faces_data), -1)  # Ensure correct shape

    # **Fix Data Length Mismatch**
    if len(faces_data) != len(names):
        print(f"⚠️ Mismatch: {len(faces_data)} face encodings vs {len(names)} names")
        
        # Keep only matching pairs
        min_length = min(len(faces_data), len(names))
        faces_data = faces_data[:min_length]
        names = names[:min_length]
        print(f"✅ Trimmed to {min_length} matching pairs.")

except Exception as e:
    print(f"Error loading face data: {e}")
    names = []
    faces_data = np.array([])
    aadhaar_data = {}

# Train KNN classifier if face data is available
knn = None
if len(faces_data) > 0 and len(names) > 0:
    knn = KNeighborsClassifier(n_neighbors=min(3, len(faces_data)), metric='euclidean')
    knn.fit(faces_data, names)
    print("✅ KNN Classifier Trained Successfully!")
else:
    print("⚠️ No registered faces found. Train the model first.")

cap = cv2.VideoCapture(0)
current_voter = {"name": "", "aadhaar": ""}

def recognize_face(frame):
    """Detects and recognizes a face using KNN continuously."""
    global current_voter

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    recognized_name = "Unknown"
    aadhaar_number = "Not Found"

    if len(faces) == 0:
        print("⚠️ No face detected!")
        return frame

    for (x, y, w, h) in faces:
        roi_gray = cv2.resize(gray[y:y+h, x:x+w], (50, 50)).flatten().reshape(1, -1)

        print(f"🔍 Checking face...")  # Debugging

        if knn is not None:
            try:
                predicted_name = knn.predict(roi_gray)[0]
                print(f"✅ Matched Name: {predicted_name}")  # Debugging

                if predicted_name in aadhaar_data:
                    aadhaar_number = aadhaar_data[predicted_name]
                    recognized_name = predicted_name
                else:
                    aadhaar_number = "Not Registered"
                    print(f"⚠️ Aadhaar not found for: {predicted_name}")

                current_voter = {"name": recognized_name, "aadhaar": aadhaar_number}
                print(f"✅ Face Matched: {recognized_name} - Aadhaar: {aadhaar_number}")

            except Exception as e:
                print(f"❌ Face Recognition Error: {e}")

        else:
            print("⚠️ KNN model not loaded!")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Aadhaar: {aadhaar_number}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame


def generate_frames():
    """Continuously streams video frames to the UI."""
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = recognize_face(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('vote.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_voter')
def get_current_voter():
    global current_voter
    return jsonify(current_voter)


@app.route('/cast_vote', methods=['POST'])
def cast_vote():
    """Stores vote securely, prevents duplicate votes, and shuts down after voting."""
    global cap
    try:
        data = request.get_json()
        print(f"📩 Received vote request: {data}")  # Debugging

        if not data or "party" not in data:
            return jsonify({"message": "❌ Error: Invalid vote request."})

        if not current_voter["name"] or current_voter["name"] == "Unknown":
            return jsonify({"message": "❌ Unrecognized voter. Cannot vote."})

        # Ensure CSV exists
        if os.path.exists("votes.csv"):
            df = pd.read_csv("votes.csv", dtype=str)
        else:
            df = pd.DataFrame(columns=["Name", "Aadhaar", "Vote", "Timestamp"])

        # Prevent multiple votes
        aadhaar_str = str(current_voter["aadhaar"])
        if aadhaar_str in df["Aadhaar"].astype(str).values:
            return jsonify({"message": "⚠️ You have already voted."})

        # Get timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Append vote
        new_vote = pd.DataFrame([[current_voter["name"], aadhaar_str, data["party"], timestamp]],
                                columns=["Name", "Aadhaar", "Vote", "Timestamp"])
        
        df = pd.concat([df, new_vote], ignore_index=True)
        df.to_csv("votes.csv", index=False)

        print(f"✅ Vote recorded: {current_voter['name']} ({aadhaar_str}) -> {data['party']}")  # Debugging

        # Close Camera
        cap.release()
        cv2.destroyAllWindows()

        # Send success message
        return jsonify({"message": "✅ Your vote has been registered! The system will close in 3 seconds."})

    except Exception as e:
        print(f"❌ Backend Error in cast_vote: {e}")  # Debugging
        return jsonify({"message": f"❌ Backend Error: {e}"})




if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True, threaded=True)