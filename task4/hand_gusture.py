# ==========================================
# TASK 04 - Hand Gesture Recognition
# (OpenCV Based - Stable Version)
# ==========================================

import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 5000:
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

            # Simple gesture logic
            if area > 20000:
                gesture = "Palm"
            else:
                gesture = "Fist"

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
