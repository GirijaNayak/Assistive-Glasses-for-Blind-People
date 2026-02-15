from ultralytics import YOLO

model = YOLO(r"C:\Users\girijanayak\Downloads\best.pt")
model.predict(source=0, conf=0.5, show=True)
