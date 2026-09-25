import cv2
import sys

VIDEO_SOURCE = 'video1.mp4'
OUTPUT_PATH = 'output.avi'
RESIZE_FACTOR = 0.6            
FPS = 10
SAVE_RESIZED = True           

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
 
    if not cap.isOpened():
        print(f'Error: could not open video source "{VIDEO_SOURCE}"')
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SAVE_RESIZED:
        out_width = int(frame_width * RESIZE_FACTOR)
        out_height = int(frame_height * RESIZE_FACTOR)
    else:
        out_width, out_height = frame_width, frame_height

    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"MJPG")
    output = cv2.VideoWriter(OUTPUT_PATH, 
                            fourcc, 
                            FPS, 
                            (out_width, out_height))

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(
                frame, 
                (0, 0), 
                fx=RESIZE_FACTOR, 
                fy=RESIZE_FACTOR,
                interpolation=cv2.INTER_AREA
            )
            output.write(resized_frame if SAVE_RESIZED else frame)

            cv2.imshow('Frame', resized_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by user.')
                break
    finally:
        cap.release()
        output.release()
        cv2.destroyAllWindows()
        print(f'Done. Processed {frame_count} frames. Saved to "{OUTPUT_PATH}".')

if __name__ == '__main__':
    main()