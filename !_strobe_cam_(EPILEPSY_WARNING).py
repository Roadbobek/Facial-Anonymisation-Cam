import numpy as np
import pyvirtualcam

flash_frame = False

with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')

    frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB

    while True:

        if flash_frame:
            frame[:] = (255, 255, 255)
        else:
            frame[:] = (0, 0, 0)

        flash_frame = not flash_frame

        cam.send(frame)
        cam.sleep_until_next_frame()
