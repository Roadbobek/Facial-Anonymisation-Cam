
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
MAX_ACTIVE_IMAGES = 10 # Max number of images on screen at once

# -----------------------------------------------------------------------------
#                            CHAOTIC EFFECT LIBRARY
# -----------------------------------------------------------------------------

# --- Static Effects (Instantaneous) ---
def static_random_flip(image):
    return cv2.flip(image, random.choice([-1, 0, 1]))

def static_rotate(image):
    return cv2.rotate(image, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]))

def static_invert_colors(image):
    return cv2.bitwise_not(image)

# --- Dynamic Effects (Evolve over time based on 'progress' from 0.0 to 1.0) ---
def dynamic_zoom(image, progress):
    h, w = image.shape[:2]
    scale = 1.0 + (progress * 2.0)
    if scale <= 0: return image
    center_x, center_y = w // 2, h // 2
    crop_w, crop_h = int(w / scale), int(h / scale)
    x1, y1 = center_x - crop_w // 2, center_y - crop_h // 2
    x2, y2 = center_x + crop_w // 2, center_y + crop_h // 2
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return cv2.resize(image, (w,h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(image[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_NEAREST)

def dynamic_blur(image, progress):
    if progress < 0.01: return image
    max_kernel = 151
    kernel_size = int(progress * max_kernel)
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 3: return image
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def dynamic_corruption(image, progress):
    img = image.copy()
    num_pixels_to_affect = int(img.size * progress * 0.1)
    for _ in range(num_pixels_to_affect):
        y, x, channel = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1), random.randint(0, 2)
        img[y, x, channel] = random.randint(0, 255)
    return img

def dynamic_fade(image, progress):
    # This fades to black. Fading *in* is handled by reversing progress.
    return cv2.addWeighted(image, 1.0 - progress, np.zeros_like(image), progress, 0)

# -----------------------------------------------------------------------------
#                               SCENE MANAGEMENT
# -----------------------------------------------------------------------------

class Scene:
    """Holds the state for a single image layer."""
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

def create_new_scene(image_files):
    """Factory for creating a new, random scene (image layer)."""
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
    duration = random.uniform(0.5, 2.5) # Each layer lasts 0.5-2.5 seconds

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
    print(f"Found {len(image_files)} images. Starting mind lobotomy v3... Press Ctrl+C to stop.")

    with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f'Using virtual camera: {cam.device}')

        active_scenes = []
        last_scene_add_time = 0
        next_add_interval = random.uniform(0.1, 0.5)

        while True:
            try:
                # --- 1. Add new scenes periodically ---
                if time.time() - last_scene_add_time > next_add_interval:
                    if len(active_scenes) < MAX_ACTIVE_IMAGES:
                        new_scene = create_new_scene(image_files)
                        if new_scene:
                            active_scenes.append(new_scene)
                    last_scene_add_time = time.time()
                    next_add_interval = random.uniform(0.1, 0.5)

                # --- 2. Prepare the canvas for this frame ---
                final_frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                if random.random() > 0.95:
                    final_frame[:] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # --- 3. Process and draw all active scenes ---
                for scene in reversed(active_scenes):
                    scene_elapsed_time = time.time() - scene.start_time

                    if scene_elapsed_time >= scene.duration:
                        active_scenes.remove(scene)
                        continue

                    progress = scene_elapsed_time / scene.duration
                    if scene.is_reversed:
                        progress = 1.0 - progress

                    frame_to_render = scene.processed_image.copy()
                    if scene.dynamic_effect:
                        frame_to_render = scene.dynamic_effect(frame_to_render, progress)

                    # --- Apply random size and position for this frame ---
                    h, w = frame_to_render.shape[:2]
                    scale_factor = random.uniform(0.2, 1.0)
                    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                    if new_w > 0 and new_h > 0:
                        frame_to_render = cv2.resize(frame_to_render, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                    # --- Place the image on the final frame ---
                    h, w = frame_to_render.shape[:2]
                    if w < WIDTH and h < HEIGHT:
                        x_pos = random.randint(0, WIDTH - w)
                        y_pos = random.randint(0, HEIGHT - h)
                        final_frame[y_pos:y_pos+h, x_pos:x_pos+w] = frame_to_render

                # --- 4. Send to virtual camera ---
                rgb_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                cam.send(rgb_frame)
                cam.sleep_until_next_frame()

            except KeyboardInterrupt:
                print("\nStopping the chaos.")
                break
            except Exception as e:
                print(f"An error occurred: {e}. Clearing scenes and continuing.")
                active_scenes.clear()
                time.sleep(0.5)

if __name__ == "__main__":
    main()
