import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening the camera")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output = cv2.VideoWriter(
    "Full_screen.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (frame_width, frame_height)
)

# Create full-screen window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Frame",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Save original frame
    output.write(frame)

    # Show frame
    cv2.imshow("Frame", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

output.release()
cap.release()
cv2.destroyAllWindows()