import os

# Add the paths to the CUDA and cuDNN bin directories for dlib, for Python 3.9 compatibility
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/13.0")
# os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/12.9")

import dlib
import cv2
import face_recognition
import pyvirtualcam
from numba import cuda
import time
import random
import colorsys




# ------------------------------------------------------ Settings ------------------------------------------------------
output_to_vcam = True # Output video to first virtual webcam found, for eg OBS.

flip_cam_input = False # Some apps, for eg discord automatically flip your camera. This is used to combat that.

flip_output_to_cam = False # Some apps, for eg discord automatically flip your camera. This is used to combat that.

use_cuda_gpu = True # False will use CPU

scale_output_to = (1920, 1080) # Resolution to scale the final image output to.

scale_window_to = scale_output_to

half_processing = False

solid_box = True
solid_box_colour = (0, 0, 0) # RGB (Red, Green, Blue) Value (Each value ranged from 0-255 so do not exceed 255)
solid_box_flashing = False # !!!!! EPILEPSY WARNING !!!!!

image_overlay = False
face_image = "Roadbobek's PFP Square.png"

face_blur = False

corruption = False
corruption_percent = 90 # Percent chance of column being coloured

corruption_double = False
corruption_double_percent = 85 # Percent Percent chance of pixel being coloured

blur_bg = False

face_label = True
face_label_text = "Roadbobek :3"
# If face_label_colour is set to "custom", it will use the RBG value specified in custom_face_label_colour
face_label_colour = "white" # [white, black, custom]
custom_face_label_colour = (63, 93, 255) # RGB (Red, Green, Blue) Value (Each value ranged from 0-255 so do not exceed 255)
colour_shift = True
colour_shift_cycle_duration = 3.0 # 3 second colour cycle

big_label = True
big_label_text = "???"
big_label_colour = (255, 255, 255) # RGB (Red, Green, Blue) Value (Each value ranged from 0-255 so do not exceed 255)
big_label_thickness = 12

fps_counter = True

title = True
title_text = "Roadbobek Cam"
# ----------------------------------------------------------------------------------------------------------------------




# --- Initialization ---
# Get a reference to webcam #0 (the default one).
video_capture = cv2.VideoCapture(0)

# Get camera resolution
cam_horz_res = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_vert_res = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Attach to virtual camera, for eg obs
if output_to_vcam:
    cam = pyvirtualcam.Camera(width=scale_output_to[0], height=scale_output_to[1], fps=30)
    print("Sending output to virtual camera.")
    print(f'Using virtual camera: {cam.device}.')

# --- Live Video Processing Variables ---
face_locations = []
# face_encodings = [] # Obsolete
more_than_one_face = False
process_this_frame = True # Process every other frame for performance.
fps = 0
reverse_0 = False
reverse_1 = False
reverse_2 = False
flash_frame = False
hue_counter = 0

# --- Load Picture/Image (s) ---
image1 = cv2.imread(face_image)

# --- Check if the image(s) was loaded successfully ---
if image1 is None:
    print(f"Error: {face_image} not found or unable to read.")
else:
    print(f"{face_image} loaded successfully!")

# --- Setup face label colour ---
if face_label_colour.casefold().strip() == "black":
    face_label_colour_rgb_final = [0, 0, 0]
elif face_label_colour.casefold().strip() == "white":
    face_label_colour_rgb_final = [255, 255, 255]
elif face_label_colour.casefold().strip() == "custom":
    face_label_colour_rgb_final = list(reversed(custom_face_label_colour)) # putText() takes a BGR value and not RBG so we reverse it for simplicity
else:
    print("---------- ERROR -- face_label_colour is wrong, please select from these three options: white, black, custom. -- ERROR ----------")

if (len(tuple(reversed(custom_face_label_colour)))) > 3:
    print("---------- ERROR -- custom_face_label_colour is wrong, (over 3 values currently specified) please input a RGB value within the parenthesis like this: (63, 93, 255). -- ERROR ----------")
elif (len(tuple(reversed(custom_face_label_colour)))) < 3:
    print("---------- ERROR -- custom_face_label_colour is wrong, (under 3 values currently specified) please input a RGB value within the parenthesis like this: (63, 93, 255). -- ERROR ----------")

print(f"dlib can use CUDA: {dlib.DLIB_USE_CUDA}")
print(f"CUDA devices: {dlib.cuda.get_num_devices()}")

if use_cuda_gpu:
    print("Using CUDA GPU.")
    print(f"Using CUDA device: {dlib.cuda.get_device()}")
    device = cuda.get_current_device()
    print(f"GPU Name: {device.name.decode('utf-8')}")
    device_context = cuda.current_context()
    device_mem = device_context.get_memory_info()
    print(f"GPU Memory (VRAM): {device_mem.total / (1024**3):.2f}GB")
    # print(f"GPU VRAM: {device.total_memory / (1024**3):.2f} GB")
else:
    print("Using CPU.")



if scale_output_to[0] == int(cam_horz_res) and scale_output_to[1] == int(cam_vert_res): # If scale_output_to and camera resolution is the same
    print(f"Not scaling output since resolution is the same: Scale Output - {scale_output_to}, Camera - ({int(cam_horz_res)}, {int(cam_vert_res)}).")
else:
    print(f"Scaling output to {scale_output_to} from ({int(cam_horz_res)}, {int(cam_vert_res)}).")

cam_fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Camera capturing at {cam_fps} FPS.")

# Create window, it is resizable but will keeps its scale. https://docs.opencv.org/3.4/d7/dfc/group__highgui.html
cv2.namedWindow("Roadbobek Cam", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

# Resize window to scale_output_to
cv2.resizeWindow("Roadbobek Cam", scale_window_to)

print(f"Window size: {scale_output_to}.")

# --- Startup Message ---
print("Starting video capture...")
print("------- Press Q to exit. -------")
# print()


program_start_time = time.time() # Get the start time of the main loop

# --- Main Video Processing Loop ---
while True:
    # start time of the loop
    start_time = time.time()

    # Get frame from webcam
    ret, frame = video_capture.read()

    # print(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # DEBUG
    # print(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # DEBUG
    # print(ret) # DEBUG
    # print() # DEBUG

    if not ret:
        print("---------- ERROR -- Failed to grab frame from webcam. Exiting... -- ERROR ----------")
        break

    # Flip incoming frame from webcam before processing.
    if flip_cam_input:
        frame = cv2.flip(frame, 2)

    # Process every other frame for performance.
    if process_this_frame:
        # Resize frame to 1/4 size for faster processing.
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR (OpenCV) to RGB (face_recognition).
        rgb_small_frame = small_frame[:, :, ::-1]

        # --- Find all faces and encodings in the current frame. ---
        if use_cuda_gpu: # GPU
            # Find all faces in the current frame using the CNN model for GPU acceleration.
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")

            # Originally Used to test if there is a face present so camera feed could be hidden if there is no face present, it is very slow. Using whats under is more efficient.
            # face_encodings = face_recognition.face_encodings(small_frame, model="cnn"). Generate face encodings using the CNN model for GPU acceleration.

            if len(face_locations) >= 1:
                more_than_one_face = True
            else:
                more_than_one_face = False

            # DEBUG
            # print(face_locations)
            # print(len(face_locations))
            # print()

        else: # CPU
            # Find all faces in the current frame using the HOG mode, utilising CPU.
            face_locations = face_recognition.face_locations(rgb_small_frame)

            # Originally Used to test if there is a face present so camera feed could be hidden if there is no face present, it is very slow. Using whats under is more efficient.
            # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            if len(face_locations) >= 1:
                more_than_one_face = True
            else:
                more_than_one_face = False


    if half_processing:
        process_this_frame = not process_this_frame


    # Get screen dimensions
    screen_height, screen_width = frame.shape[:2]

    # --- Blur Background ---
    if blur_bg:
        ksize = (16, 16)
        frame = cv2.blur(frame, ksize, cv2.BORDER_DEFAULT)


    # DEBUG
    # if more_than_one_face:
    #     print("Faces present.")
    #     # print(top, right, bottom, left)
    # else:
    #     print("No faces present.")
    #     screen_height, screen_width = frame.shape[:2]
    #     print(frame.shape)
    #     cv2.rectangle(img=frame, pt1=(0, 0), pt2=(screen_width, screen_height), color=(0, 0, 0), lineType=cv2.FILLED)

    if more_than_one_face: # Used to use face_encodings
        # --- Display Results on Original Frame ---
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since detection was on 1/4 size.
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # --- DEBUG ---
            # print("Faces present.")
            # print(face_locations)
            # # print(face_encodings) # Obsolete
            # print(top, right, bottom, left)


            ## --- Box drawing example ---
            # pt1 is the top-left corner of the rectangle (x1, y1)
            # pt2 is the bottom-right corner of the rectangle (x2, y2)
            # In this case, (left, top) is pt1 and (right, bottom) is pt2.
            # cv2.rectangle(img=frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 0), thickness=2, lineType=cv2.FILLED)

            # --- Draw a solid box (-1 thickness) ---
            if solid_box:
                if flash_frame:
                    solid_box_colour = (255, 255, 255)
                else:
                    solid_box_colour = (0, 0, 0)

                cv2.rectangle(img=frame, pt1=(left -100, top -100), pt2=(right + 100, bottom + 100), color=solid_box_colour, thickness=-1)
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2, cv2.FILLED)

                if solid_box_flashing:
                    flash_frame = not flash_frame




            # --- Drawing new image to current image (frame) ---
            if image_overlay:
                # this is wrong this draws image to a window
                # whats the difference if i use the top line? I guess you can resize it and it already exists.
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Default: cv2.WINDOW_AUTOSIZE, with WINDOW_NORMAL we can resize it.
                # cv2.imshow('image', image1) # This is what the code is already doing at the end so we need to add our new image over the video frame image (frame)

                # print() # DEBUG
                # print(f"frame size: {frame.size}") # DEBUG
                # print(f"frame shape: {frame.shape}") # DEBUG
                # print() # DEBUG
                # print(f"image1 size: {image1.size}") # DEBUG
                # print(f"image1 shape: {image1.shape}") # DEBUG
                # print() # DEBUG
                # print(left -90, top -90) # DEBUG - top left pixel of the face cover in current window
                # print(right + 90, bottom + 90) # DEBUG - bottom right pixel of the face cover in current window
                # print() # DEBUG
                # print((left -90) - (right + 90)) # DEBUG - shape of face cover PxP
                # print((top -90) - (bottom + 90)) # DEBUG - shape of face cover PxP
                # print() # DEBUG

                new_top = top if top > 100 else 100
                new_left = left if left > 100 else 100
                # new_bottom = bottom if bottom < cam_vert_res else int(cam_vert_res)
                # new_right = right if right < cam_horz_res else int(cam_horz_res)

                # Size of face cover
                face_cover_size = frame[new_top - 100:bottom + 100, new_left - 100:right + 100].shape
                face_cover_size_bottom_right = face_cover_size[1]
                face_cover_size_top_left = face_cover_size[0]

                # face_cover_size_top_left = abs((new_top -90) - (bottom + 90))
                # face_cover_size_bottom_right = abs((new_left -90) - (right + 90))

                # print(frame[new_top - 90:bottom + 90, new_left - 90:right + 90].shape) # DEBUG
                # print(face_cover_size_bottom_right) # DEBUG
                # print(face_cover_size_top_left) # DEBUG
                # print() # DEBUG

                image1_resize = cv2.resize(image1, (face_cover_size_bottom_right, face_cover_size_top_left)) # Interpolation classes: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121

                # print(image1.shape) # DEBUG
                # print(image1_resize.shape) # DEBUG
                # print() # DEBUG

                # print(face_cover_size_top_left) # DEBUG
                # print(face_cover_size_bottom_right) # DEBUG
                # print("-----") # DEBUG
                # print(top) # DEBUG
                # print(bottom) # DEBUG
                # print(left) # DEBUG
                # print(right) # DEBUG
                # print() # DEBUG
                # print() # DEBUG

                # print(frame[new_top - 90:bottom + 90, new_left - 90:right + 90].shape) # DEBUG
                # print(top, bottom, left, right) # DEBUG
                # print(new_top, bottom, new_left, right) # DEBUG
                # print(image1_resize.shape) # DEBUG
                # print(face_cover_size_top_left) # DEBUG
                # print(face_cover_size_bottom_right) # DEBUG
                # print() # DEBUG

                frame[new_top - 100:bottom + 100, new_left - 100:right + 100] = image1_resize

                # Blurring here will result in the picture being blurred aswell
                # if blur_bg:
                #     ksize = (16, 16)
                #     frame = cv2.blur(frame, ksize, cv2.BORDER_DEFAULT)



            # --- Blur face ---
            if face_blur:
                # print("-----") # DEBUG
                # print(top) # DEBUG
                # print(bottom) # DEBUG
                # print(left) # DEBUG
                # print(right) # DEBUG
                # print() # DEBUG

                new_top = top if top > 100 else 100
                new_left = left if left > 100 else 100

                # print(new_top) # DEBUG
                # print(bottom) # DEBUG
                # print(new_left) # DEBUG
                # print(right) # DEBUG
                # print("-----") # DEBUG

                # Extract the region of the image that contains the face
                blurred_face = frame[new_top - 100:bottom + 100, new_left - 100:right + 100]
                # Blur the face image
                blurred_face = cv2.GaussianBlur(blurred_face, (139, 139), 50)
                # Put the blurred face region back into the frame image
                frame[new_top - 100:bottom + 100, new_left - 100:right + 100] = blurred_face

                # frame[top - 100:bottom + 100, left - 100:right + 100] = cv2.GaussianBlur(frame[top - 100:bottom + 100, left - 100:right + 100], (151, 151), 50)

                # # Extract the region of the image that contains the face
                # face_image = frame[top:bottom, left:right]
                # # Blur the face image
                # face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
                # # Put the blurred face region back into the frame image
                # frame[top:bottom, left:right] = face_image


            # --- Corruption ---
            if corruption:
                # print("-----") # DEBUG
                # print(top) # DEBUG
                # print(bottom) # DEBUG
                # print(left) # DEBUG
                # print(right) # DEBUG
                # print() # DEBUG

                new_top = top if top > 100 else 100
                new_left = left if left > 100 else 100

                # print(new_top) # DEBUG
                # print(bottom) # DEBUG
                # print(new_left) # DEBUG
                # print(right) # DEBUG
                # print("-----") # DEBUG

                # Extract the region of the image that contains the face
                face = frame[new_top - 100:bottom + 100, new_left - 100:right + 100]

                for i in face:
                    # print(f"len(i): {print(len(i))}")
                    # print(i) # DEBUG
                    # print() # DEBUG

                    colour_this_pixel1 = random.randint(1, 100)

                    if colour_this_pixel1 <= corruption_percent:
                        i[0:] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

                    # print(i) # DEBUG
                    # print("-----------------------") # DEBUG
                    # print() # DEBUG


            # --- Double corruption ---
            if corruption_double:
                # print("-----") # DEBUG
                # print(top) # DEBUG
                # print(bottom) # DEBUG
                # print(left) # DEBUG
                # print(right) # DEBUG
                # print() # DEBUG

                new_top = top if top > 100 else 100
                new_left = left if left > 100 else 100

                # print(new_top) # DEBUG
                # print(bottom) # DEBUG
                # print(new_left) # DEBUG
                # print(right) # DEBUG
                # print("-----") # DEBUG

                # Extract the region of the image that contains the face
                face = frame[new_top - 100:bottom + 100, new_left - 100:right + 100]

                for i in face:
                    # print(f"len(i): {print(len(i))}")
                    # print(i) # DEBUG
                    # print() # DEBUG

                    for ii in i:
                        colour_this_pixel2 = random.randint(1, 100)

                        if colour_this_pixel2 <= corruption_double_percent:
                            ii[0:] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

                    # print(i) # DEBUG
                    # print("-----------------------") # DEBUG
                    # print() # DEBUG


            # --- Big label ---
            if big_label:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = big_label_text
                text_color = big_label_colour
                text_thickness = big_label_thickness

                # Define the face_cover rectangle dimensions
                cover_left = left - 90
                cover_top = top - 90
                cover_right = right + 90
                cover_bottom = bottom + 90

                cover_width = cover_right - cover_left
                cover_height = cover_bottom - cover_top

                # Ensure cover dimensions are positive to avoid issues
                if cover_width > 0 and cover_height > 0:
                    # Get the size of the text with a font scale of 1.0
                    (base_text_width, base_text_height), _ = cv2.getTextSize(text, font, 1.0, text_thickness)

                    if base_text_width > 0 and base_text_height > 0:
                        # Calculate the optimal font scale to fit the text within the cover
                        # Add a little padding so text isn't flush against the cover edges
                        padding_factor = 0.8  # Text will take up to 80% of the cover dimension

                        scale_for_width = (cover_width * padding_factor) / base_text_width
                        scale_for_height = (cover_height * padding_factor) / base_text_height

                        optimal_font_scale = min(scale_for_width, scale_for_height)
                        # Ensure font scale is not excessively small or zero
                        optimal_font_scale = max(optimal_font_scale, 0.1)

                        # Get the final text size with the optimal scale
                        (final_text_width, final_text_height), _ = cv2.getTextSize(text, font, optimal_font_scale, text_thickness)

                        # Calculate the center of the original face (which is also the center of the cover)
                        center_x_face = (left + right) // 2
                        center_y_face = (top + bottom) // 2

                        # Calculate text position to center it in the cover
                        # (text_x, text_y) is the bottom-left corner for cv2.putText
                        text_x = center_x_face - final_text_width // 2
                        text_y = center_y_face + final_text_height // 2 # Vertically centers the text

                        # Draw the scaled text
                        cv2.putText(frame, text, (text_x, text_y), font, optimal_font_scale, text_color, text_thickness)
                    else:
                        # Fallback or skip if base text size is zero (e.g., empty text string)
                        pass # Or draw with a default small size if preferred
                else:
                    # Fallback if cover dimensions are not positive (should not happen with detected faces)
                    pass



            # --- Draw label ---
            if face_label:

                # cover_width = (right + 50) - (left - 50)
                # text_width = cover_width / 375
                #
                # # print(face_cover_size_bottom_right / 425.0) # DEBUG
                # print(cover_width) # DEBUG
                # print(text_width) # DEBUG
                # # print() # DEBUG

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, face_label_text, (left - 50, bottom + 50), font, 1, (face_label_colour_rgb_final), 2) # cv2.putText(frame, face_label_text, (left - 50, bottom + 50), font, text_width, (face_label_colour_rgb_final), 2)

                # cv2.putText(frame, face_label_text, (left - 30, bottom + 45), font, 1.0, (255, 255, 255), 1)
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, "Anonymous", (left + 6, bottom - 9), font, 0.9, (255, 255, 255), 1)

                # Colour shift, just cause


                if colour_shift:
                    # 1. Calculate the elapsed time and use it to get a looping hue value
                    elapsed_time = time.time() - program_start_time
                    h = (elapsed_time % colour_shift_cycle_duration) / colour_shift_cycle_duration
                    s = 1.0 # Saturation to full
                    v = 1.0 # Value to full

                    # 2. Convert HSV to RGB (0.0 to 1.0)
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)

                    # 3. Scale and convert RGB to BGR for OpenCV
                    bgr_color = (b * 255, g * 255, r * 255)

                    # 4. Update the color for your text
                    face_label_colour_rgb_final = bgr_color

                    # print(face_label_colour_rgb_final) # DEBUG
                    # print(type(face_label_colour_rgb_final)) # DEBUG
                    # print() # DEBUG

                    # for index, i in enumerate(face_label_colour_rgb_final):
                    #     if index == 0:
                    #         if reverse_0 == False and face_label_colour_rgb_final[index] >= 255:
                    #             reverse_0 = True
                    #         elif reverse_0 != False and face_label_colour_rgb_final[index] <= 0:
                    #             reverse_0 = False
                    #
                    #         if reverse_0:
                    #             face_label_colour_rgb_final[index] -= 2 # random.randint(0, 5)
                    #         else:
                    #             face_label_colour_rgb_final[index] += 2 # random.randint(0, 5)
                    #
                    #
                    #     if index == 1:
                    #         if reverse_1 == False and face_label_colour_rgb_final[index] >= 255:
                    #             reverse_1 = True
                    #         elif reverse_1 != False and face_label_colour_rgb_final[index] <= 0:
                    #             reverse_1 = False
                    #
                    #         if reverse_1:
                    #             face_label_colour_rgb_final[index] -= 2 # random.randint(0, 5)
                    #         else:
                    #             face_label_colour_rgb_final[index] += 2 # random.randint(0, 5)
                    #
                    #
                    #     if index == 2:
                    #         if reverse_2 == False and face_label_colour_rgb_final[index] >= 255:
                    #             reverse_2 = True
                    #         elif reverse_2 != False and face_label_colour_rgb_final[index] <= 0:
                    #             reverse_2 = False
                    #
                    #         if reverse_2:
                    #             face_label_colour_rgb_final[index] -= random.randint(0, 14)
                    #         else:
                    #             face_label_colour_rgb_final[index] += random.randint(0, 14)


                        # print(f"index: {index}") # DEBUG
                        # print(face_label_colour_rgb_final[index]) # DEBUG
                        # print(type(face_label_colour_rgb_final[index])) # DEBUG
                        # print() # DEBUG

                    # print(f"reverse_0: {reverse_0}") # DEBUG
                    # print(f"reverse_1: {reverse_1}") # DEBUG
                    # print(f"reverse_2: {reverse_2}") # DEBUG
                    # print(face_label_colour_rgb_final) # DEBUG
                    # print("------------------------------") # DEBUG
                    # print() # DEBUG


    # --- FPS counter ---
    if fps_counter:
        fps_horz_pos = (scale_output_to[0] // 22) # 15 before title
        fps_vert_pos = (scale_output_to[0] // 12) # 17 before title

        # print(fps_horz_pos) # DEBUG
        # print(fps_vert_pos) # DEBUG
        # print() # DEBUG

        cv2.putText(frame, f"FPS: {fps:.2f}", (fps_horz_pos, fps_vert_pos), cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 40, 40), 2)

    fps = (1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

    if fps >= cam_fps:
        fps = 30



    # --- Title text ---
    if title:
        title_horz_pos = (scale_output_to[0] // 22)
        title_vert_pos = (scale_output_to[0] // 17)

    # print(title_horz_pos) # DEBUG
    # print(title_vert_pos) # DEBUG
    # print() # DEBUG

    cv2.putText(frame, title_text, (title_horz_pos, title_vert_pos), cv2.FONT_HERSHEY_DUPLEX, 2.0, (40, 40, 40), 4)


    # --- If no face recognised block view ---
    if not more_than_one_face: # Used to use face_encodings
        # DEBUG
        # print("No faces present.")
        # print(frame.shape)

        cv2.rectangle(img=frame, pt1=(0, 0), pt2=(screen_width, screen_height), thickness=-1, color=(0, 0, 0))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "0.o", (screen_width // 2 - 350, screen_height // 2 + 100), font, 14, (255, 255, 255), 22)



    # Scale frame resolution if different one compared to camera res is specified
    if not (scale_output_to[0] == int(cam_horz_res) and scale_output_to[1] == int(cam_vert_res)):
        frame = cv2.resize(frame, scale_output_to) # Scale the frame image.


    # Display the resulting image in our window.
    cv2.imshow('Roadbobek Cam', frame)

    # Flip frame before outputing to virtual camera
    if flip_output_to_cam:
        frame = cv2.flip(frame, 2)

    # Display frame in virtual webcam, for eg OBS virtual camera
    if output_to_vcam:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam.send(rgb_frame)



    # Detect input
    keypress = cv2.waitKey(1)

    # if keypres != -1: # DEBUG
    #     print(keypres) # DEBUG

    # Hit 'q' on the keyboard to quit
    if keypress & 0xFF == ord('q') or cv2.getWindowProperty("Roadbobek Cam", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Save last frame to png when 'e' is pressed
    if keypress & 0xFF == ord('e'):
        cv2.imwrite("frame.png", frame)

    # Resize window to scale_output_to when 'r' is pressed
    if keypress & 0xff == ord('r'):
        cv2.resizeWindow("Roadbobek Cam", scale_window_to)

    # # Hit 'q' on the keyboard to quit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    #
    # # Save last frame to png when 'e' is pressed
    # if cv2.waitKey(1) & 0XFF == ord('e'):
    #     cv2.imwrite("frame.png", frame)


# --- Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
