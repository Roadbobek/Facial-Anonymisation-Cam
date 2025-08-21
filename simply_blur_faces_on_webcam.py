import os

# Add the paths to the CUDA and cuDNN bin directories for dlib, for Python 3.9 compatibility
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64")
os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/13.0")
# os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/12.9")

import face_recognition
import cv2
import numpy

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract the region of the image that contains the face
        face_image = frame[top - 100:bottom + 100, left - 100:right + 100]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        # face_image = cv2.stackBlur(face_image, (99, 99))
        # face_image = cv2.blur(face_image, (99, 99))
        # face_image = cv2.medianBlur(face_image, 99) # ):

        # Put the blurred face region back into the frame image
        frame[top - 100:bottom + 100, left - 100:right + 100] = face_image

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
