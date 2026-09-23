import cv2

image_path = "image3.png"

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Resize to 60%
resized = cv2.resize(
    image,
    None,
    fx=0.6,
    fy=0.6,
    interpolation=cv2.INTER_AREA
)
cv2.imshow("Original Image", image)
cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()