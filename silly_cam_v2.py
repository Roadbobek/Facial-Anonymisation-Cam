
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

# -----------------------------------------------------------------------------
#                            CHAOTIC EFFECT LIBRARY
# -----------------------------------------------------------------------------

# --- Static Effects (Instantaneous) ---

def static_random_flip(image):
    """Randomly flips the image horizontally, vertically, or both."""
    return cv2.flip(image, random.choice([-1, 0, 1]))

def static_rotate(image):
    """Rotates the image by a random 90-degree increment."""
    return cv2.rotate(image, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]))

def static_invert_colors(image):
    """Inverts the colors of the image."""
    return cv2.bitwise_not(image)

# --- Dynamic Effects (Evolve over time based on 'progress' from 0.0 to 1.0) ---

def dynamic_zoom(image, progress):
    """Zooms into or out of the center of the image."""
    h, w = image.shape[:2]
    # Go from 1x scale to 3x scale based on progress
    scale = 1.0 + (progress * 2.0)
    if scale <= 0: return image # Failsafe

    # Get the center of the image
    center_x, center_y = w // 2, h // 2

    # Define the box to crop
    crop_w, crop_h = int(w / scale), int(h / scale)
    x1 = center_x - crop_w // 2
    y1 = center_y - crop_h // 2
    x2 = center_x + crop_w // 2
    y2 = center_y + crop_h // 2

    # Failsafe for rounding errors
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)

    # Crop and resize back to original dimensions
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

def dynamic_blur(image, progress):
    """Applies a blur that gets stronger over time."""
    if progress < 0.01: return image # No blur at the start
    # Kernel size must be an odd number
    max_kernel = 151
    kernel_size = int(progress * max_kernel)
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 3: return image # Failsafe
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def dynamic_corruption(image, progress):
    """Randomly messes up an increasing percentage of color channels."""
    img = image.copy()
    num_pixels_to_affect = int(img.size * progress * 0.1) # Corrupt up to 10% of all pixel data

    for _ in range(num_pixels_to_affect):
        y = random.randint(0, img.shape[0] - 1)
        x = random.randint(0, img.shape[1] - 1)
        channel = random.randint(0, 2)
        img[y, x, channel] = random.randint(0, 255)
    return img

def dynamic_fade(image, progress):
    """Fades the image to black."""
    return cv2.addWeighted(image, 1.0 - progress, np.zeros_like(image), progress, 0)

# -----------------------------------------------------------------------------
#                               SCENE MANAGEMENT
# -----------------------------------------------------------------------------

class Scene:
    """Holds the state for a single chaotic scene."""
    def __init__(self, base_image, static_effects, dynamic_effect, is_reversed, duration):
        self.base_image = base_image
        self.static_effects = static_effects
        self.dynamic_effect = dynamic_effect
        self.is_reversed = is_reversed # Does the dynamic effect go 0->1 or 1->0?
        self.duration = duration
        self.start_time = time.time()

        # Pre-apply static effects so we don't do it every frame
        self.processed_image = self.base_image.copy()
        for effect in self.static_effects:
            self.processed_image = effect(self.processed_image)

def create_new_scene(image_files):
    """Factory for creating a new, random scene."""
    # 1. Pick a random image
    image_path = random.choice(image_files)
    base_image = cv2.imread(image_path)
    if base_image is None: return None # Failsafe

    # 2. Decide on the scene type (static, dynamic, or both)
    scene_type = random.choice(['static', 'dynamic', 'both'])

    static_effects_pool = [static_random_flip, static_rotate, static_invert_colors]
    dynamic_effects_pool = [dynamic_zoom, dynamic_blur, dynamic_corruption, dynamic_fade]

    static_effects = []
    dynamic_effect = None

    if scene_type == 'static':
        static_effects.append(random.choice(static_effects_pool))
    elif scene_type == 'dynamic':
        dynamic_effect = random.choice(dynamic_effects_pool)
    elif scene_type == 'both':
        static_effects.append(random.choice(static_effects_pool))
        dynamic_effect = random.choice(dynamic_effects_pool)

    # 3. Decide other parameters
    is_reversed = random.choice([True, False]) # e.g., zoom in vs zoom out
    duration = random.uniform(0.05, 0.3) # Scene lasts 0.05-0.3 seconds

    # print(f"New Scene: {os.path.basename(image_path)}, Type: {scene_type}, Dynamic: {dynamic_effect.__name__ if dynamic_effect else 'None'}, Reversed: {is_reversed}, Duration: {duration:.2f}s")

    return Scene(base_image, static_effects, dynamic_effect, is_reversed, duration)

# -----------------------------------------------------------------------------
#                                 MAIN PROGRAM
# -----------------------------------------------------------------------------

def main():
    """The main function to run the chaotic virtual camera."""
    # --- Initialization ---
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        print(f"--- ERROR: The folder '{IMAGE_FOLDER}' does not exist or is empty. ---")
        # Placeholder error message logic remains the same...
        return

    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)]
    print(f"Found {len(image_files)} images. Starting mind lobotomy... Press Ctrl+C to stop.")

    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f'Using virtual camera: {cam.device}')

        current_scene = create_new_scene(image_files)
        if not current_scene: return # Exit if first image fails

        while True:
            try:
                # --- Check if it's time for a new scene ---
                scene_elapsed_time = time.time() - current_scene.start_time
                if scene_elapsed_time >= current_scene.duration:
                    current_scene = create_new_scene(image_files)
                    if not current_scene: continue
                    scene_elapsed_time = 0

                # --- Calculate effect progress (0.0 to 1.0) ---
                progress = scene_elapsed_time / current_scene.duration
                if current_scene.is_reversed:
                    progress = 1.0 - progress

                # --- Render the frame ---
                # Start with the pre-processed image (with static effects)
                frame_to_render = current_scene.processed_image.copy()

                # Apply the dynamic effect if there is one
                if current_scene.dynamic_effect:
                    frame_to_render = current_scene.dynamic_effect(frame_to_render, progress)

                # --- Place the final image on the black canvas ---
                # Create a background canvas, with a chance of a random color
                final_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                if random.random() > 0.8:
                    final_frame[:] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

                # Resize to a new random size for this specific frame
                h, w = frame_to_render.shape[:2]
                scale_factor = random.uniform(0.25, 1.5)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                if new_w > 0 and new_h > 0:
                    frame_to_render = cv2.resize(frame_to_render, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Place the randomly-sized image at a random position
                h, w = frame_to_render.shape[:2]
                if w < WIDTH and h < HEIGHT:
                    x_pos = random.randint(0, WIDTH - w)
                    y_pos = random.randint(0, HEIGHT - h)
                    final_frame[y_pos:y_pos+h, x_pos:x_pos+w] = frame_to_render
                else:
                    # If it's too big, just resize it to fill the screen
                    final_frame = cv2.resize(frame_to_render, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

                # --- Send to virtual camera ---
                rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)

                # Wait until the next frame is due, for smooth animation.
                cam.sleep_until_next_frame()

            except KeyboardInterrupt:
                print("\nStopping the chaos.")
                break
            except Exception as e:
                print(f"An error occurred: {e}. Creating a new scene.")
                current_scene = create_new_scene(image_files) # Try to recover by starting fresh
                if not current_scene:
                    print("Could not recover. Exiting.")
                    break
                time.sleep(0.5)

if __name__ == "__main__":
    main()
