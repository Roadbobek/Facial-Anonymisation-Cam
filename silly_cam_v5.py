
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
    scale = 1.0 + (progress * 2.0)
    if scale <= 0: return image # Failsafe
    center_x, center_y = w // 2, h // 2
    crop_w, crop_h = int(w / scale), int(h / scale)
    x1, y1 = center_x - crop_w // 2, center_y - crop_h // 2
    x2, y2 = center_x + crop_w // 2, center_y + crop_h // 2
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

def dynamic_blur(image, progress):
    """Applies a blur that gets stronger over time."""
    if progress < 0.01: return image
    max_kernel = 151
    kernel_size = int(progress * max_kernel)
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 3: return image
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def dynamic_corruption(image, progress):
    """Randomly messes up an increasing percentage of color channels."""
    img = image.copy()
    num_pixels_to_affect = int(img.size * progress * 0.1)
    for _ in range(num_pixels_to_affect):
        y, x, channel = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1), random.randint(0, 2)
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
        self.is_reversed = is_reversed
        self.duration = duration
        self.start_time = time.time()

        # Pre-apply static effects
        self.processed_image = self.base_image.copy()
        for effect in self.static_effects:
            self.processed_image = effect(self.processed_image)

        # Set a fixed size and position for the entire scene's duration
        h, w = self.processed_image.shape[:2]
        scale_factor = random.uniform(0.25, 1.5)
        self.scaled_w, self.scaled_h = int(w * scale_factor), int(h * scale_factor)

        if self.scaled_w < WIDTH and self.scaled_h < HEIGHT and self.scaled_w > 0 and self.scaled_h > 0:
            self.x_pos = random.randint(0, WIDTH - self.scaled_w)
            self.y_pos = random.randint(0, HEIGHT - self.scaled_h)
        else:
            # If it's too big or small, just have it fill the screen
            self.scaled_w, self.scaled_h = WIDTH, HEIGHT
            self.x_pos, self.y_pos = 0, 0

def create_new_scene(image_files):
    """Factory for creating a new, random scene."""
    image_path = random.choice(image_files)
    base_image = cv2.imread(image_path)
    if base_image is None: return None

    scene_type = random.choice(['static', 'dynamic', 'both'])
    static_effects_pool = [static_random_flip, static_rotate, static_invert_colors]
    dynamic_effects_pool = [dynamic_zoom, dynamic_blur, dynamic_corruption, dynamic_fade]
    static_effects, dynamic_effect = [], None

    if scene_type == 'static':
        static_effects.append(random.choice(static_effects_pool))
    elif scene_type == 'dynamic':
        dynamic_effect = random.choice(dynamic_effects_pool)
    elif scene_type == 'both':
        static_effects.append(random.choice(static_effects_pool))
        dynamic_effect = random.choice(dynamic_effects_pool)

    is_reversed = random.choice([True, False])
    duration = random.uniform(0.05, 0.2)
    return Scene(base_image, static_effects, dynamic_effect, is_reversed, duration)

# -----------------------------------------------------------------------------
#                                 MAIN PROGRAM
# -----------------------------------------------------------------------------

def main():
    """The main function to run the chaotic virtual camera."""
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        print(f"--- ERROR: The folder '{IMAGE_FOLDER}' does not exist or is empty. ---")
        return

    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)]
    print(f"Found {len(image_files)} images. Starting mind lobotomy... Press Ctrl+C to stop.")

    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f'Using virtual camera: {cam.device}')
        current_scene = create_new_scene(image_files)
        if not current_scene: return

        while True:
            try:
                scene_elapsed_time = time.time() - current_scene.start_time
                if scene_elapsed_time >= current_scene.duration:
                    current_scene = create_new_scene(image_files)
                    if not current_scene: continue

                progress = (time.time() - current_scene.start_time) / current_scene.duration
                if current_scene.is_reversed: progress = 1.0 - progress

                frame_to_render = current_scene.processed_image.copy()
                if current_scene.dynamic_effect:
                    frame_to_render = current_scene.dynamic_effect(frame_to_render, progress)

                final_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                if random.random() > 0.95:
                    final_frame[:] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

                # Resize the (potentially dynamically altered) image to the size defined for this scene
                resized_image = cv2.resize(frame_to_render, (current_scene.scaled_w, current_scene.scaled_h), interpolation=cv2.INTER_NEAREST)

                # Place it at the position defined for this scene using robust bounds checking
                x, y, w, h = current_scene.x_pos, current_scene.y_pos, current_scene.scaled_w, current_scene.scaled_h
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w, WIDTH), min(y + h, HEIGHT)
                src_x1, src_y1 = x1 - x, y1 - y
                src_x2, src_y2 = x2 - x, y2 - y

                if src_x2 > src_x1 and src_y2 > src_y1:
                    final_frame[y1:y2, x1:x2] = resized_image[src_y1:src_y2, src_x1:src_x2]

                rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()

            except KeyboardInterrupt:
                print("\nStopping the chaos.")
                break
            except Exception as e:
                # print(f"An error occurred: {e}. Creating a new scene.") # Optional: for debugging
                current_scene = create_new_scene(image_files)
                if not current_scene:
                    print("Could not recover. Exiting.")
                    break

if __name__ == "__main__":
    main()
