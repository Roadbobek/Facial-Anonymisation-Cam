import cv2
import numpy as np
import time

# --- Setup and Initialization ---

# Path to the image file.
# NOTE: Replace 'your_image.jpg' with the path to your own image file.
# For example: 'path/to/my_image.png'
# If the image is in the same directory as the script, just use its filename.
image_path = "weird.png"

# Create a dummy image if the placeholder is not found, so the script can still run.
# This prevents the program from crashing if the user doesn't have an image ready.
try:
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Using a dummy image instead.")
except FileNotFoundError as e:
    print(e)
    # Create a 400x400 black image with a blue rectangle and white text
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), -1)
    cv2.putText(image, "Dummy Image", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.namedWindow("Rotating Image", flags=cv2.WINDOW_NORMAL)

# Get the dimensions (height and width) of the image
(h, w) = image.shape[:2]
# Calculate the center of the image
(cX, cY) = (w // 2, h // 2)

# Set the initial rotation angle
angle = 0

# The rotation speed in degrees per frame
rotation_speed = 0.75


# --- Main Rotation Loop ---

print("Press 'q' to quit the program.")

while True:
    # Create the rotation matrix
    # cv2.getRotationMatrix2D(center, angle, scale)
    # The matrix rotates the image around its center without scaling
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Apply the rotation using the affine transform
    # cv2.warpAffine(source_image, rotation_matrix, destination_size)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Display the rotated image in a window
    cv2.imshow("Rotating Image", rotated_image)

    # Increment the angle for the next frame
    angle = (angle + rotation_speed) % 360

    # Wait for a key press. `cv2.waitKey(1)` waits for 1 millisecond.
    # This creates the 'per frame' delay.
    # If the 'q' key is pressed, break the loop.
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# --- Cleanup ---

# Destroy all OpenCV windows
cv2.destroyAllWindows()

