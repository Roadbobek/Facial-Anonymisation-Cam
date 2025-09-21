
import os
import cv2
import numpy as np
import pyvirtualcam
import random
import time

# --- Configuration ---
IMAGE_FOLDER = "silly_images"
WIDTH, HEIGHT = 1280, 720
FPS = 60
MAX_IMAGES_ON_SCREEN = 15
SPAWN_INTERVAL = 0.2  # Seconds between new images appearing

# -----------------------------------------------------------------------------
#                            CHAOTIC EFFECT LIBRARY
# -----------------------------------------------------------------------------

def effect_warp(image):
    """Stretches and squashes the image to a random aspect ratio."""
    h, w = image.shape[:2]
    new_w = random.randint(w // 4, w * 2)
    new_h = random.randint(h // 4, h * 2)
    if new_w <= 0 or new_h <= 0: return image
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def effect_extreme_blur(image):
    """Applies a very strong blur effect."""
    kernel_size = random.randrange(31, 101, 2)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def effect_corrupt_colors(image):
    """Randomly messes up the color channels of the image."""
    img = image.copy()
    channel_to_corrupt = random.randint(0, 2)
    img[:, :, channel_to_corrupt] = np.random.randint(0, 256, size=img.shape[:2], dtype=np.uint8)
    return img

def effect_flip(image):
    """Randomly flips the image horizontally, vertically, or both."""
    return cv2.flip(image, random.choice([-1, 0, 1]))

def effect_rotate(image):
    """Rotates the image by a random 90-degree increment."""
    return cv2.rotate(image, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]))

# -----------------------------------------------------------------------------
#                           IMAGE OBJECT MANAGEMENT
# -----------------------------------------------------------------------------

class SillyImage:
    """Manages a single image, its effects, movement, and lifetime."""
    def __init__(self, image):
        # --- Apply a random permanent effect --- 
        effects = [effect_warp, effect_extreme_blur, effect_corrupt_colors, effect_flip, effect_rotate, lambda x: x]
        chosen_effect = random.choice(effects)
        self.original_image = chosen_effect(image)

        # --- Movement and Scaling --- 
        self.x, self.y = random.randint(0, WIDTH), random.randint(0, HEIGHT)
        self.scale = random.uniform(0.1, 0.8)
        self.move_speed_x = random.uniform(-600, 1200) # Pixels per second
        self.move_speed_y = random.uniform(-600, 1200)
        self.scale_speed = random.uniform(-1.0, 1.6) # Scale units per second
        self.lifetime = random.uniform(1.0, 6.0) # Seconds

    def update(self, delta_time):
        """Updates position, scale, and lifetime. Returns False if dead."""
        self.x += self.move_speed_x * delta_time
        self.y += self.move_speed_y * delta_time
        self.scale += self.scale_speed * delta_time

        # Bounce off edges
        if self.x < 0 or self.x > WIDTH: self.move_speed_x *= -1
        if self.y < 0 or self.y > HEIGHT: self.move_speed_y *= -1
        if self.scale < 0.1 or self.scale > 2.0: self.scale_speed *= -1
        self.scale = np.clip(self.scale, 0.05, 2.5) # Keep scale in a reasonable range

        self.lifetime -= delta_time
        return self.lifetime > 0

    def draw(self, frame):
        """Draws the image onto the main frame with robust bounds checking."""
        try:
            h, w = self.original_image.shape[:2]
            scaled_w, scaled_h = int(w * self.scale), int(h * self.scale)
            if scaled_w <= 0 or scaled_h <= 0: return

            processed_image = cv2.resize(self.original_image, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

            # Top-left corner of the image
            x_start, y_start = int(self.x - scaled_w / 2), int(self.y - scaled_h / 2)
            
            # --- Bounds Checking to Fix Crash ---
            # Find the overlapping region between the image and the frame
            x1 = max(x_start, 0)
            y1 = max(y_start, 0)
            x2 = min(x_start + scaled_w, WIDTH)
            y2 = min(y_start + scaled_h, HEIGHT)

            # Calculate the corresponding region in the source image
            src_x1 = x1 - x_start
            src_y1 = y1 - y_start
            src_x2 = x2 - x_start
            src_y2 = y2 - y_start

            # If there is a valid, non-empty region to draw
            if src_x2 > src_x1 and src_y2 > src_y1:
                frame[y1:y2, x1:x2] = processed_image[src_y1:src_y2, src_x1:src_x2]
        except Exception:
            pass # Ignore errors during drawing, which can happen with extreme scales

# -----------------------------------------------------------------------------
#                                 MAIN PROGRAM
# -----------------------------------------------------------------------------

def main():
    """The main function to run the chaotic virtual camera."""
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        print(f"--- ERROR: The folder '{IMAGE_FOLDER}' does not exist or is empty. ---")
        return

    image_files = [cv2.imread(os.path.join(IMAGE_FOLDER, f)) for f in os.listdir(IMAGE_FOLDER)]
    image_files = [img for img in image_files if img is not None] # Filter out failed loads
    print(f"Found and loaded {len(image_files)} images. Starting mind lobotomy v3... Press Ctrl+C to stop.")

    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f'Using virtual camera: {cam.device}')
        
        silly_images = []
        last_spawn_time = time.time()
        last_frame_time = time.time()

        while True:
            try:
                # --- Delta Time Calculation ---
                current_time = time.time()
                delta_time = current_time - last_frame_time
                last_frame_time = current_time

                # --- Spawn New Images ---
                if current_time - last_spawn_time > SPAWN_INTERVAL and len(silly_images) < MAX_IMAGES_ON_SCREEN:
                    silly_images.append(SillyImage(random.choice(image_files)))
                    last_spawn_time = current_time

                # --- Update and Draw ---
                final_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                
                # Update existing images and remove dead ones
                alive_images = []
                for silly_image in silly_images:
                    if silly_image.update(delta_time):
                        alive_images.append(silly_image)
                silly_images = alive_images

                # Draw the images
                for silly_image in silly_images:
                    silly_image.draw(final_frame)

                # --- Send to Camera ---
                rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()

            except KeyboardInterrupt:
                print("\nStopping the chaos.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(0.5)

if __name__ == "__main__":
    main()
