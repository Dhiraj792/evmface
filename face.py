import cv2
import pickle
import numpy as np
import os

# Ensure 'data' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Load existing Aadhaar data if available
aadhaar_data = {}
if os.path.exists('data/aadhaar_data.pkl'):
    with open('data/aadhaar_data.pkl', 'rb') as f:
        aadhaar_data = pickle.load(f)

aadhaar_number = input("Enter your Aadhaar number: ")

# Check if Aadhaar is already registered
if aadhaar_number in aadhaar_data:
    print("⚠️ Aadhaar number already exists! Face not registered.")
else:
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces_data = []
    framesTotal = 30  # Reduce frame count for faster training

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            crop_img = gray[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten()
            faces_data.append(resized_img)

            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= framesTotal:
            break

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.array(faces_data).reshape(len(faces_data), -1)

    # Load existing names if available
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
    else:
        names = []

    names.append(aadhaar_number)  # Store Aadhaar as the identifier
    aadhaar_data[aadhaar_number] = aadhaar_number  # Ensure Aadhaar is linked correctly

    # Save data
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
    with open('data/aadhaar_data.pkl', 'wb') as f:
        pickle.dump(aadhaar_data, f)

    print("✅ Face registration successful! Aadhaar saved.")
