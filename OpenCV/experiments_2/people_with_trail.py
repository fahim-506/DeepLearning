                                                 # CPU 

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict, deque

# model = YOLO('yolov8n.pt')
# cap = cv2.VideoCapture('../street.mp4')

# id_map = {}
# nex_id = 1

# trail = defaultdict(lambda: deque(maxlen=30))
# appear = defaultdict(int)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     res = model.track(frame, classes=[0], persist=True, verbose=False)
#     annotated_frame = frame.copy()

#     if res[0].boxes.id is not None:
#         boxes = res[0].boxes.xyxy.cpu().numpy()
#         ids = res[0].boxes.id.cpu().numpy()

#         for box, oid in zip(boxes, ids):
#             x1, y1, x2, y2 = map(int, box)
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#             appear[oid] += 1

#             if appear[oid] >= 5 and oid not in id_map:
#                 id_map[oid] = nex_id
#                 nex_id += 1

#             if oid in id_map:
#                 sid = id_map[oid]
#                 trail[oid].append((cx, cy))

#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(
#                     annotated_frame,
#                     f'ID: {sid}',
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 0, 255),
#                     2
#                 )
#                 cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

#     cv2.imshow('Tracking', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

                                                        # GPU

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque

# Check MPS availability
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and move to device
model = YOLO('yolov8n.pt')
model.to(device)

cap = cv2.VideoCapture('../street.mp4')

id_map = {}
next_id = 1

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    res = model.track(frame, classes=[0], persist=True, verbose=False, device=device)
    annotated_frame = frame.copy()

    # Safe check for detections
    if res and res[0].boxes is not None and res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Count appearances correctly
            appear[oid] += 1

            # Assign stable ID after 5 frames
            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw ID
                cv2.putText(
                    annotated_frame,
                    f'ID: {sid}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # Draw center
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

                # Draw trail
                for i in range(1, len(trail[oid])):
                    cv2.line(
                        annotated_frame,
                        trail[oid][i - 1],
                        trail[oid][i],
                        (0, 255, 255),
                        2
                    )

    cv2.imshow('Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()