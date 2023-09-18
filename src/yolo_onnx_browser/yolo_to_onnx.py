from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trainednpm install

# Export the model
model.export(format='onnx')