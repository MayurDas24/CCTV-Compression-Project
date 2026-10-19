import cv2
import time
from pathlib import Path

output = Path("live_test/webcam_30s.mp4")
fps = 20.0
duration = 30.0
interval = 1.0 / fps

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open webcam.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise SystemExit("ERROR: Could not create MP4 output.")

count = 0
start = time.monotonic()
next_frame = start
deadline = start + duration

try:
    print(f"Recording for {duration:.0f}s at {fps:.0f} FPS: {width}x{height}")
    while time.monotonic() < deadline:
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        if time.monotonic() >= deadline:
            break

        ok, frame = cap.read()
        if not ok:
            print("WARNING: Webcam frame read failed.")
            break

        writer.write(frame)
        count += 1
        next_frame += interval
finally:
    cap.release()
    writer.release()

elapsed = time.monotonic() - start
print(f"Frames written: {count}")
print(f"Elapsed seconds: {elapsed:.2f}")
print(f"Output: {output.resolve()}")

if count == 0:
    raise SystemExit("ERROR: No frames recorded.")
