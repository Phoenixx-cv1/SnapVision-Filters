import cv2
import numpy as np


# Modes
PREVIEW = 0
BLUR = 1
FEATURES = 2
CANNY = 3
BEAUTY = 4
SKETCH = 5
SEPIA = 6
GLITCH = 7
SUNGLASSES = 8

# Load sunglasses overlay (PNG with alpha)
overlay = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Helper: apply sunglasses overlay on face
def apply_overlay(frame, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, int(h / 3)))
    for i in range(overlay_resized.shape[0]):
        for j in range(overlay_resized.shape[1]):
            if overlay_resized[i, j][3] != 0:  # Alpha channel
                frame[y + i, x + j] = overlay_resized[i, j][:3]
    return frame

# Helper: apply filters
def beauty_filter(frame):
    blur = cv2.bilateralFilter(frame, 9, 75, 75)
    return cv2.addWeighted(frame, 0.6, blur, 0.4, 0)

def sketch_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def sepia_filter(frame):
    sepia = np.array(frame, dtype=np.float64)
    sepia = cv2.transform(sepia, np.matrix([[0.272, 0.534, 0.131],
                                            [0.349, 0.686, 0.168],
                                            [0.393, 0.769, 0.189]]))
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def rgb_glitch(frame):
    b, g, r = cv2.split(frame)
    r = np.roll(r, 5, axis=1)
    g = np.roll(g, -5, axis=0)
    return cv2.merge([b, g, r])

# Main
image_filter = PREVIEW
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)
alive = True
win_name = "SnapVision - Real-Time Fun Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

video_path="person.mp4"

cap = cv2.VideoCapture(video_path)

frame_size= (int(cap.get(3)),int(cap.get(4)))
out=cv2.VideoWriter("filters_output_person.mp4",cv2.VideoWriter_fourcc(*'mpv4'),20.0,frame_size)
    

while alive:
    has_frame, frame = cap.read()
    if not has_frame:
        break
        
    result = frame.copy()
    

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, **feature_params)
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 6, (0, 255, 255), -1)
    elif image_filter == BEAUTY:
        result = beauty_filter(frame)
    elif image_filter == SKETCH:
        result = sketch_filter(frame)
    elif image_filter == SEPIA:
        result = sepia_filter(frame)
    elif image_filter == GLITCH:
        result = rgb_glitch(frame)
    elif image_filter == SUNGLASSES:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        result = frame
        for (x, y, w, h) in faces:
            try:
                result = apply_overlay(result, overlay, x, y + int(h/5), w, h)
            except:
                pass

    cv2.imshow(win_name, result)
    out.write(result)

    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        alive = False
    elif key == ord("p"): image_filter = PREVIEW
    elif key == ord("b"): image_filter = BLUR
    elif key == ord("c"): image_filter = CANNY
    elif key == ord("f"): image_filter = FEATURES
    elif key == ord("1"): image_filter = BEAUTY
    elif key == ord("2"): image_filter = SKETCH
    elif key == ord("3"): image_filter = SEPIA
    elif key == ord("4"): image_filter = GLITCH
    elif key == ord("5"): image_filter = SUNGLASSES

cap.release()
out.release()
cv2.destroyAllWindows()