import logging
from ultralytics import YOLO

# Suppress logging messages
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Model
model = YOLO("hat-model.pt")


results = model.predict("tester.mp4", imgsz=320, conf=0.9)

# Check for detections
if any(len(result) > 0 for result in results):
    print(True)
else:
    print(False)
