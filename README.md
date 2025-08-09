# SnapVision - Real-Time Filters
An OpenCV Python project that applies various real-time video filters on a video file including blur, sketch, sepia, glitch, and fun sunglasses overlay.

Features:
- Apply multiple filters on video frames:
- Preview (no filter)
- Blur
- Edge detection (Canny)
- Feature point detection
- Beauty filter (bilateral blur)
- Sketch effect
- Sepia tone
- RGB glitch effect
- Sunglasses overlay on detected faces using Haar cascades

Reads video file and writes filtered output video.

Real-time filter switching using keyboard.
Quit with q or ESC

# Requirements
- Python 3.x
- OpenCV(opencv-python)
- Numpy
Install dependencies:
pip install opencv-python numpy

# Usage
- Place your input video file (e.g.,person.mp4) in the project directory.
- Make sure sunglass.png (transparent PNG with alpha channel) is present.
- Run the script: python snapvision_filters.py
- Use keyboard keys to toggle filters.
- The filtered video will be saved as filters_output_person.mp4.
# Demo Video
[Watch the output video here](https://drive.google.com/drive/folders/1gxJyuV6mW1PSDkQ7xDXzcVCW-YK1QjYq?usp=drive_link)
  
