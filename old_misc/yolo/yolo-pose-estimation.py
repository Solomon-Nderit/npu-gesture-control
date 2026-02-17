from ultralytics import YOLO

#Loading the model
model=YOLO("yolo11n-pose.pt")

results=model.track(source="people-walking.mp4", show=True)