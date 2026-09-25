import cv2
import math
import sys
from pathlib import Path
from typing import cast
from ultralytics import YOLO
from ultralytics.engine.results import Results

VIDEO_PATH = 'video1.mp4'
MODEL_PATH = 'yolov8n.pt'
OUTPUT_PATH = 'output.avi'
CONF_THRESHOLD = 0.4
DISPLAY_RESIZE_FACTOR = 0.6
FPS = 10

def main():
    video_path = Path(VIDEO_PATH)
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        print(f'Error: model file not found at "{model_path}"')
        sys.exit(1)
    if not video_path.exists():
        print(f'Error: video file not found at "{video_path}"')
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print('Error: could not open the video.')
        sys.exit(1)

    model = YOLO(str(model_path))
    class_names = model.names

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"MJPG")
    output = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (frame_width, frame_height))

    frame_count = 0
    detection_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = cast(list[Results], model(frame, conf=CONF_THRESHOLD, verbose=False))

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = class_names[cls]
                    label = f'{class_name} {conf}'
                    detection_count += 1

                    text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + text_size[0], y1 - text_size[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)

            output.write(frame)

            resized_frame = cv2.resize(
                frame, (0, 0), fx=DISPLAY_RESIZE_FACTOR, fy=DISPLAY_RESIZE_FACTOR,
                interpolation=cv2.INTER_AREA
            )
            cv2.imshow('frame', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by user.')
                break
    finally:
        cap.release()
        output.release()
        cv2.destroyAllWindows()
        print(f'Done. Processed {frame_count} frames, '
              f'{detection_count} total detections. Saved to "{OUTPUT_PATH}".')

if __name__ == '__main__':
    main()