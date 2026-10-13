import cv2
import json
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Load class names in the same order used during training
with open("class_names.json", "r") as file:
    class_names = json.load(file)

# Simple smiley representations supported by OpenCV
smileys = {
    "angry": ">:(",
    "disgust": ":/",
    "fear": ":o",
    "happy": ":)",
    "neutral": ":|",
    "sad": ":(",
    "surprise": ":O"
}

# Load OpenCV's face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load the face detector.")

# Start webcam: 0 usually means the default camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

print("Webcam started. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            # Extract face region
            face = gray[y:y + h, x:x + w]

            # Preprocess to match CNN input
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            # Predict emotion
            probabilities = model.predict(
                face, verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            emotion = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])

            # Draw face rectangle
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            # Draw emotion and smiley
            label = (
                f"{emotion.upper()} "
                f"{smileys.get(emotion, ':)')} "
                f"{confidence * 100:.1f}%"
            )

            text_y = max(y - 10, 25)

            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow(
            "CNN Real-Time Emotion Recognition",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()