import cv2
import sys
from pathlib import Path
from typing import cast
from ultralytics import YOLO
from ultralytics.engine.results import Results

MODEL_PATH = 'yolov8n.pt'
IMAGE_PATH = 'image3.png'
OUTPUT_PATH = 'output.png'
CONF_THRESHOLD = 0.25
SHOW_WINDOW = True   # set False if running headless (no display available)

def main():
    model_path = Path(MODEL_PATH)
    image_path = Path(IMAGE_PATH)


    if not model_path.exists():
        print(f'Error: model file not found at "{model_path}"')
        sys.exit(1)
    if not image_path.exists():
        print(f'Error: image file not found at "{image_path}"')
        sys.exit(1)

    model = YOLO(str(model_path))


    results = cast(list[Results], model(str(image_path), conf=CONF_THRESHOLD, verbose=False))
    result = results[0]

    annotated = result.plot()

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print('No objects detected above the confidence threshold.')
    else:
        names = result.names
        detected = [names[int(cls)] for cls in boxes.cls]
        print(f'Detected {len(detected)} object(s): {", ".join(detected)}')
        
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f'Annotated image saved to "{OUTPUT_PATH}"')

    if SHOW_WINDOW:
        try:
            cv2.imshow('Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print('Display window unavailable (headless environment) — '
                  'result was still saved to disk.')

if __name__ == '__main__':
    main()