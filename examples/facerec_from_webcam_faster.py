import face_recognition
import cv2
import numpy as np
import sys

# --- Initialization ---
# Get a reference to webcam #0 (the default one).
video_capture = cv2.VideoCapture(0)

# --- Load and Encode Known Faces ---
# This section loads your reference images and computes their face encodings.

# --- Face 1 Processing ---
print("Processing face 1...")
face1_image = face_recognition.load_image_file("face5_no_exif.jpg")
face1_encodings = face_recognition.face_encodings(face1_image)

# Ensure a face was found in the image.
if len(face1_encodings) > 0:
    face1_encoding = face1_encodings[0]
    print("Face 1 processed successfully.")
else:
    print("Error: No face found in face5_no_exif.jpg. Please check the image and try again.")
    sys.exit(1) # Exit if a crucial reference face isn't found.


# --- Face 2 Processing ---
print("Processing face 2...")
face2_image = face_recognition.load_image_file("obama.jpg")
face2_encodings = face_recognition.face_encodings(face2_image)

# Ensure a face was found in the image.
if len(face2_encodings) > 0:
    face2_encoding = face2_encodings[0]
    print("Face 2 processed successfully.")
else:
    print("Error: No face found in obama.jpg. Please check the image and try again.")
    sys.exit(1) # Exit if a crucial reference face isn't found.


# --- Known Faces Database ---
# Create arrays of known face encodings and their corresponding names.
# The order here must match the order of `known_face_encodings`.
known_face_encodings = [
    face1_encoding,
    face2_encoding
]
known_face_names = [
    "Lucas Gard",
    "Bomboclat"
]

# --- Live Video Processing Variables ---
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True # Process every other frame for performance.

# --- Startup Message ---
print("Starting video capture...")
print("Press Q to exit.")

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

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s).
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face.
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # --- Display Results on Original Frame ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since detection was on 1/4 size.
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image.
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
video_capture.release()
cv2.destroyAllWindows()