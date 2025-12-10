import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# =====================================
# 1. MODEL LOADING
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded ✅")

# =====================================
# 2. DIGIT → ACTION MAPPING
# =====================================
digit_actions = {
    0: "STOP",
    1: "MOVE FORWARD (Slow)",
    2: "MOVE FORWARD (Medium)",
    3: "MOVE FORWARD (Fast)",
    4: "TURN LEFT (Small)",
}

# Color coding for actions
action_colors = {
    0: (0, 100, 255),      # Red for STOP
    1: (0, 255, 100),      # Green for Slow
    2: (0, 255, 200),      # Light green for Medium
    3: (0, 200, 255),      # Yellow-green for Fast
    4: (255, 150, 0),      # Blue for Turn
}

CONF_THRESHOLD = 0.5

# =====================================
# 3. HELPER FUNCTIONS FOR UI
# =====================================
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=15):
    """Draw a rectangle with rounded corners"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    if thickness < 0:  # Filled
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_confidence_bar(img, x, y, width, height, confidence):
    """Draw a confidence meter bar"""
    # Background
    draw_rounded_rect(img, (x, y), (x + width, y + height), (50, 50, 50), -1, 5)
    
    # Foreground (confidence level)
    fill_width = int(width * confidence)
    if fill_width > 0:
        color = (0, int(255 * (1 - confidence)), int(255 * confidence))
        draw_rounded_rect(img, (x, y), (x + fill_width, y + height), color, -1, 5)

# =====================================
# 4. WEBCAM SETUP
# =====================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

cv2.namedWindow("Digit Detection", cv2.WINDOW_NORMAL)

last_digit = None
prev_time = time.time()
fps_smooth = 0.0

print("🎥 Webcam started. Press 'q' to quit.")

# =====================================
# 5. MAIN LOOP
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # -----------------------------
    # 5a. Inference
    # -----------------------------
    results = model(frame, verbose=False)
    r = results[0]
    boxes = r.boxes

    detected_digit = None
    detected_conf = 0.0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Track best detection
            if conf > detected_conf:
                detected_conf = conf
                detected_digit = cls

            # Draw bounding box with action-specific color
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            box_color = action_colors.get(cls, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            # Corner accents for modern look
            corner_len = 20
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, 5)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, 5)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, 5)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, 5)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, 5)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, 5)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, 5)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, 5)

            # Label with rounded background
            label = f"Digit {cls}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            draw_rounded_rect(frame, (x1, y1 - th - 20), (x1 + tw + 20, y1 - 5), box_color, -1, 8)
            cv2.putText(frame, label, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -----------------------------
    # 5b. Top Control Panel
    # -----------------------------
    panel_height = 120
    overlay = frame.copy()
    draw_rounded_rect(overlay, (15, 15), (w - 15, panel_height), (25, 25, 25), -1, 15)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Status indicator circle
    status_x = 40
    status_y = 50
    if detected_digit is not None and detected_conf >= CONF_THRESHOLD:
        status_color = action_colors.get(detected_digit, (0, 255, 0))
        cv2.circle(frame, (status_x, status_y), 15, status_color, -1)
        cv2.circle(frame, (status_x, status_y), 18, status_color, 2)
        
        # Action text
        action_str = digit_actions.get(detected_digit, "UNKNOWN")
        cv2.putText(frame, f"DIGIT {detected_digit}", (status_x + 35, status_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, action_str, (status_x + 35, status_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
        
        # Confidence bar
        draw_confidence_bar(frame, status_x + 35, status_y + 30, 250, 15, detected_conf)
        cv2.putText(frame, f"{detected_conf*100:.1f}%", (status_x + 295, status_y + 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        last_digit = detected_digit
    else:
        # Idle state
        cv2.circle(frame, (status_x, status_y), 15, (100, 100, 100), -1)
        cv2.circle(frame, (status_x, status_y), 18, (150, 150, 150), 2)
        cv2.putText(frame, "SEARCHING...", (status_x + 35, status_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        last_digit = None

    # -----------------------------
    # 5c. Bottom Info Bar
    # -----------------------------
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0
    fps_smooth = fps_smooth * 0.9 + fps * 0.1  # Smooth FPS
    prev_time = cur_time

    info_overlay = frame.copy()
    draw_rounded_rect(info_overlay, (15, h - 55), (w - 15, h - 15), (25, 25, 25), -1, 10)
    frame = cv2.addWeighted(info_overlay, 0.85, frame, 0.15, 0)

    # FPS and threshold info
    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (30, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence Threshold: {CONF_THRESHOLD:.0%}", (200, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    
    # Instructions
    cv2.putText(frame, "Press 'Q' to quit", (w - 200, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Digit Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exit.")