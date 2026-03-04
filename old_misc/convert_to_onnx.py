from ultralytics import YOLO

model = YOLO("../models/hand-trained.pt")

model.export(format="onnx")