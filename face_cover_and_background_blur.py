import face_recognition
import cv2


# --- Settings ---
blur_bg = True
cutey_mode = False

# --- Initialization ---
# Get a reference to webcam #0 (the default one).
video_capture = cv2.VideoCapture(0)

# --- Live Video Processing Variables ---
face_locations = []
face_encodings = []
process_this_frame = True # Process every other frame for performance.

# --- Startup Message ---
print("Starting video capture...")
print("---- Press Q to exit. ---")

# --- Main Video Processing Loop ---
while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame from webcam. Exiting...")
        break

    # Process every other frame for performance.
    if process_this_frame:
        # Resize frame to 1/4 size for faster processing.
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR (OpenCV) to RGB (face_recognition).
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all faces and encodings in the current frame.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


    process_this_frame = not process_this_frame


    # Get screen dimensions
    screen_height, screen_width = frame.shape[:2]

    # --- Blur Everything ---
    if blur_bg:
        ksize = (16, 16)
        frame = cv2.blur(frame, ksize, cv2.BORDER_DEFAULT)


    # if face_encodings:
    #     print("Faces present.")
    #     # print(top, right, bottom, left)
    # else:
    #     print("No faces present.")
    #     screen_height, screen_width = frame.shape[:2]
    #     print(frame.shape)
    #     cv2.rectangle(img=frame, pt1=(0, 0), pt2=(screen_width, screen_height), color=(0, 0, 0), lineType=cv2.FILLED)

    if face_encodings:
        # --- Display Results on Original Frame ---
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since detection was on 1/4 size.
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # DEBUG
            # print("Faces present.")
            # print(face_locations)
            # print(face_encodings)
            # print(top, right, bottom, left)


            ## Box drawing example
            # pt1 is the top-left corner of the rectangle (x1, y1)
            # pt2 is the bottom-right corner of the rectangle (x2, y2)
            # In this case, (left, top) is pt1 and (right, bottom) is pt2.
            # cv2.rectangle(img=frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 0), thickness=2, lineType=cv2.FILLED)

            # Draw a solid box (-1 thickness)
            cv2.rectangle(img=frame, pt1=(left -90, top -90), pt2=(right + 90, bottom + 90), color=(0, 0, 0), thickness=-1)
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2, cv2.FILLED)

            # # Draw label
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, "Anonymous", (left - 30, bottom + 45), font, 1.0, (255, 255, 255), 1)
            ## cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
            ## font = cv2.FONT_HERSHEY_DUPLEX
            ## cv2.putText(frame, "Anonymous", (left + 6, bottom - 9), font, 0.9, (255, 255, 255), 1)


            # --- Cutey mode! >v< ---
            if cutey_mode:
                font = cv2.FONT_HERSHEY_DUPLEX
                text = ">v<"
                text_color = (255, 255, 255)
                text_thickness = 2

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


    # --- If no face recognised block view ---
    if not face_encodings:
        # DEBUG
        # print("No faces present.")
        # print(frame.shape)

        cv2.rectangle(img=frame, pt1=(0, 0), pt2=(screen_width, screen_height), thickness=-1, color=(0, 0, 0))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "xD", (screen_width // 2, screen_height // 2), font, 3, (255, 255, 255), 2)


    # Display the resulting image.
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit.w
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --- Cleanup ---
video_capture.release()
cv2.destroyAllWindows()
