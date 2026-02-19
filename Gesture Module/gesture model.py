import cv2
import numpy as np
import tensorflow as tf
import json


model = tf.keras.models.load_model("gesture_mobilenet_hagrid.keras")


with open("gesture_classes.json", "r") as f:
    class_indices = json.load(f)


class_names = {v: k for k, v in class_indices.items()}
num_classes = len(class_names)

print("Loaded classes:", class_names)


IMG_SIZE = 224

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    img = preprocess_frame(frame)


    preds = model.predict(img, verbose=0)

   
    if preds.shape[1] != num_classes:
        gesture = "Invalid output"
    else:
        pred_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        gesture = class_names.get(pred_index, "Unknown")

   
    cv2.putText(
        frame,
        f"Gesture: {gesture} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
