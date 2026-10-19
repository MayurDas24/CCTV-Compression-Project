
import cv2
import time
from pathlib import Path

output = Path("live_test/webcam_5s.mp4")
output.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 20.0

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

frames = 0
start = time.monotonic()
next_frame = start

try:
    print("Webcam recording started. Look at the camera for 5 seconds.")
    while time.monotonic() - start < 5:
        ok, frame = cap.read()
        if not ok:
            print("Warning: webcam frame could not be read.")
            break

        writer.write(frame)
        frames += 1

        next_frame += 1.0 / fps
        time.sleep(max(0, next_frame - time.monotonic()))
finally:
    writer.release()
    cap.release()

elapsed = time.monotonic() - start
print(f"Recording finished: {output.resolve()}")
print(f"Frames captured: {frames}")
print(f"Elapsed seconds: {elapsed:.2f}")
print(f"File size: {output.stat().st_size:,} bytes")
