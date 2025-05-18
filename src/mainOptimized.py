import numpy as np
import cv2
import threading
from keras.models import load_model

# Load model
model = load_model("../models/finalModelOptimized.h5")

# Constants
ROI_top, ROI_bottom = 100, 300
ROI_right, ROI_left = 150, 350
accumulated_weight = 0.5
word_dict = {
    0: 'One', 1: 'Ten', 2: 'Two', 3: 'Three', 4: 'Four',
    5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'
}

# Globals for threading
background = None
frame_lock = threading.Lock()
frame = None
result_text = ""
num_frames = 0

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return (thresholded, max(contours, key=cv2.contourArea))

def prediction_loop():
    global frame, result_text, num_frames
    while True:
        with frame_lock:
            local_frame = frame.copy() if frame is not None else None
        if local_frame is None:
            continue

        roi = local_frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if num_frames < 70:
            cal_accum_avg(gray, accumulated_weight)
            result_text = "Calibrating..."
        else:
            hand = segment_hand(gray)
            if hand:
                thresholded, _ = hand
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1, 64, 64, 3))
                pred = model.predict(thresholded, verbose=0)
                result_text = word_dict[np.argmax(pred)]
            else:
                result_text = ""

# Start the prediction thread
threading.Thread(target=prediction_loop, daemon=True).start()

# Video capture loop
cam = cv2.VideoCapture(0)
while True:
    ret, frm = cam.read()
    if not ret:
        break
    frm = cv2.flip(frm, 1)
    with frame_lock:
        frame = frm
    num_frames += 1

    display_frame = frm.copy()
    cv2.rectangle(display_frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    if result_text:
        cv2.putText(display_frame, result_text, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(display_frame, "Real-time Hand Sign Detection", (10, 20), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)

    cv2.imshow("Sign Detection", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
