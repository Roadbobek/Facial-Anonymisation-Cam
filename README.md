# *Face Cover And Background Blur*

### This project provides a proof-of-concept for real-time facial anonymisation, achieved through face covering and background blurring. It is built using Python, leveraging the [face_recognition](https://github.com/ageitgey/face_recognition) library for its straightforward implementation of facial detection and landmarks, alongside opencv-python for image manipulation.

#### It's important to note that this proof of concept prioritizes ease of development and clarity of demonstration over raw performance. Due to Python's interpreted nature and the overhead of the face_recognition library (which is built on top of dlib and face_recognition_models), this solution is inherently slower and more resource-intensive than one directly optimized with lower-level OpenCV operations or implemented in a compiled language. While effective for demonstrating the core functionality, users should expect higher processing times, especially with high-resolution video streams.

#### Author: Roadbobek

#### Version: 1.0.0

## Installation
I made this project on Windows 11, the face_recognition library doesn't officially support Windows so the process was very hard.

If you are using Windows please refer to this Youtube video: <br>
https://youtu.be/xaDJ5xnc8dc.

if you are using Linux or macOS please refer to the face_recognition installation instructions: <br>
https://github.com/ageitgey/face_recognition.
### Requirements

- Python 3.8.2
  - face_recognition
      - (pip module, in CMD/Terminal run 'pip install face_recognition'.)
  - cv2
      - (pip module, in CMD/Terminal run 'pip install opencv-python'.)
- All face_recognition requirements
  - https://github.com/ageitgey/face_recognition

## License

Copyright (c) 2025 Roadbobek

This software is licensed under the MIT License.

See the LICENSE.txt file for details

#### MIT License