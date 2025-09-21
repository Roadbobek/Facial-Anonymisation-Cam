
import os
import cv2
import numpy as np
import pyvirtualcam
import random
import time

# --- Configuration ---
IMAGE_FOLDER = "silly_images"
WIDTH, HEIGHT = 1280, 720
FPS = 30

# --- Helper Functions for Chaotic Effects ---

def corrupt_colors(image):
    """Randomly messes up the color channels of the image."""
    img = image.copy()
    channel_to_corrupt = random.randint(0, 2)
    img[:, :, channel_to_corrupt] = np.random.randint(0, 256, size=img.shape[:2], dtype=np.uint8)
    return img

def random_flip(image):
    """Randomly flips the image horizontally, vertically, or both."""
    flip_code = random.choice([-1, 0, 1])
    return cv2.flip(image, flip_code)

def extreme_blur(image):
    """Applies a very strong blur effect."""
    kernel_size = random.randrange(51, 201, 2) # Random odd kernel size
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def warp_image(image):
    """Stretches and squashes the image to a random aspect ratio."""
    h, w = image.shape[:2]
    new_w = random.randint(w // 4, w * 2)
    new_h = random.randint(h // 4, h * 2)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def rotate_image(image):
    """Rotates the image by a random 90-degree increment."""
    return cv2.rotate(image, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]))

def main():
    """The main function to run the chaotic virtual camera."""
    # --- Initialization ---
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        print(f"--- ERROR ---")
        print(f"The folder '{IMAGE_FOLDER}' does not exist or is empty.")
        print(f"Please create the folder and add some images to it.")
        # Create a placeholder frame to show the error
        error_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(error_frame, f"Folder '{IMAGE_FOLDER}' not found", (50, HEIGHT // 2 - 30), font, 1.5, (0, 0, 255), 3)
        cv2.putText(error_frame, "Please create it and add images.", (50, HEIGHT // 2 + 30), font, 1.5, (0, 0, 255), 3)

        # Send the error frame to the camera
        with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
            print(f'Using virtual camera: {cam.device}')
            while True:
                cam.send(cv2.cvtColor(error_frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
        return

    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)]
    print(f"Found {len(image_files)} images in '{IMAGE_FOLDER}'.")

    # List of all possible chaotic effects
    effects = [
        corrupt_colors,
        random_flip,
        extreme_blur,
        warp_image,
        rotate_image,
        lambda img: cv2.bitwise_not(img), # Invert colors
        lambda img: np.random.permutation(img) # Scramble rows
    ]

    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f'Using virtual camera: {cam.device}')
        print("Starting mind lobotomy... Press Ctrl+C to stop.")

        while True:
            try:
                # --- Create a masterpiece of chaos for each frame ---

                # 1. Pick a random image
                random_image_path = random.choice(image_files)
                image = cv2.imread(random_image_path)
                if image is None:
                    continue # Skip if the image is invalid

                # 2. Apply a random number of random effects
                num_effects_to_apply = random.randint(1, 3)
                for _ in range(num_effects_to_apply):
                    chosen_effect = random.choice(effects)
                    image = chosen_effect(image)

                # 3. Create a background canvas
                final_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

                # Add a random flashing background color
                if random.random() > 0.8:
                    final_frame[:] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

                # 4. Place the mangled image randomly on the canvas
                h, w = image.shape[:2]
                max_x = WIDTH - w
                max_y = HEIGHT - h

                if max_x > 0 and max_y > 0:
                    x_pos = random.randint(0, max_x)
                    y_pos = random.randint(0, max_y)
                    final_frame[y_pos:y_pos+h, x_pos:x_pos+w] = image
                else:
                    # If the image is too big, just resize it to fit
                    final_frame = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

                # 5. Send to virtual camera
                # Convert from OpenCV's BGR format to RGB for the camera
                rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)

                # 6. 1 in 20 chance of large pause
                pause = random.randint(1, 20)

                # 7. Random large pause
                if pause == 1:
                    time.sleep(random.uniform(0.3, 0.8))
                else:
                    # 8. Regular pause, wait for a very short, random amount of time
                    time.sleep(random.uniform(0.025, 0.2))

            except KeyboardInterrupt:
                print("\nStopping the chaos.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                # On error, just show a black screen for a moment
                time.sleep(0.5)


if __name__ == "__main__":
    main()
