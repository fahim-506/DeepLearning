import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Device setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign'
]

# Video
cap = cv2.VideoCapture('../street.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img).to(device)

    # Inference
    with torch.no_grad():
        outputs = model([img_tensor])

    annotated_frame = frame.copy()

    # Process detections
    for box, label, score in zip(outputs[0]['boxes'],
                                 outputs[0]['labels'],
                                 outputs[0]['scores']):

        if score > 0.5 and label == 1:  # class 1 = person
            x1, y1, x2, y2 = map(int, box.cpu().numpy())

            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(annotated_frame,
                        f'Person {score:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    cv2.imshow("Faster R-CNN Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()