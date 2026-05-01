import cv2
import numpy as np
import math
import csv
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

cap = cv2.VideoCapture(0)

id_counter = 0
objects = {}

angle = 0

# Create CSV log
log_file = open("tracking_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["Frame", "ID", "Class", "X", "Y", "Speed"])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # ================= YOLO DETECTION =================
    results = model(frame, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append((cx, cy, x1, y1, x2, y2, label))

    new_objects = {}

    # ================= TRACKING =================
    for (cx, cy, x1, y1, x2, y2, label) in detections:

        matched_id = None

        for obj_id, points_list in objects.items():
            px, py = points_list[-1]

            distance = ((cx - px)**2 + (cy - py)**2) ** 0.5

            if distance < 50:
                matched_id = obj_id
                break

        # MATCHED
        if matched_id is not None:
            new_objects[matched_id] = objects[matched_id] + [(cx, cy)]
            points = new_objects[matched_id]
            obj_id = matched_id

        # NEW OBJECT
        else:
            new_objects[id_counter] = [(cx, cy)]
            points = new_objects[id_counter]
            obj_id = id_counter
            id_counter += 1

        # ================= TRAJECTORY =================
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0,255,255), 2)

        # ================= PREDICTION =================
        speed = 0
        if len(points) >= 5:
            vx_total = 0
            vy_total = 0

            for i in range(1, 5):
                x_prev, y_prev = points[-i-1]
                x_curr, y_curr = points[-i]

                vx_total += (x_curr - x_prev)
                vy_total += (y_curr - y_prev)

            vx = vx_total / 4
            vy = vy_total / 4

            pred_x = int(points[-1][0] + vx)
            pred_y = int(points[-1][1] + vy)

            cv2.circle(frame, (pred_x, pred_y), 6, (0,0,255), -1)

            speed = (vx**2 + vy**2) ** 0.5

        # ================= THREAT =================
        if speed > 15:
            threat = "HIGH"
            color = (0,0,255)
        elif speed > 5:
            threat = "MEDIUM"
            color = (0,255,255)
        else:
            threat = "LOW"
            color = (0,255,0)

        # ================= DRAW =================
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} ID {obj_id} [{threat}]",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ================= LOGGING =================
        writer.writerow([frame_count, obj_id, label, cx, cy, speed])

    objects = new_objects

    # ================= DISPLAY =================
    cv2.imshow("YOLO Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()