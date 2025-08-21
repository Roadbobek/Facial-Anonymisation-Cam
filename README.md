# *Facial Anonymisation Cam*

### This project provides real-time facial anonymisation. It is built using Python, leveraging the [face_recognition](https://github.com/ageitgey/face_recognition) library for its straightforward implementation of facial detection and landmarks, alongside opencv-python for image manipulation. This project can run on CUDA GPUs for enhanced performance.

#### It's important to note that this solution prioritizes ease of development and clarity of demonstration over raw performance. Due to Python's interpreted nature and the overhead of the face_recognition library (which is built on top of dlib and face_recognition_models), this program is inherently slower and more resource-intensive than one directly optimized with lower-level OpenCV operations or implemented in a compiled language. While effective for demonstrating the core functionality, users should expect higher processing times, especially with high-resolution video streams.

#### Author: Roadbobek

#### Version: 2.0.0

---

### Demonstration Video

[![Demo Video](https://img.youtube.com/vi/6wMMOnadwXo/hqdefault.jpg)](https://www.youtube.com/watch?v=6wMMOnadwXo)

### ***Click Me! ^^^***

---

## Installation

This project was built and tested on Windows 11.

The installation process for the `dlib` library, which is a core dependency, can be complex, especially for GPU support.

The facial_recognition library does not have official Windows support, meaning it was a long and difficult process to get everything working. I honestly do not remember most of the things I did to get everything to work, for eg to get dlib built with CUDA support I had to edit the source code to compile for my GPUs compute architecture. For these reasons I cannot really provide a good tutorial that will work for everyone.

### Prerequisites

You need to have the following installed on your system:
- **Python 3.9.13** (recommended version)
- **Visual Studio** with the **Desktop development with C++** workload selected.
- **CMake**

For GPU installation, you will also need:
- **NVIDIA GPU Drivers**
- **NVIDIA CUDA Toolkit**
- **cuDNN**

### Python Libraries

The Python libraries needed depend on your installation method.

#### For CPU Installation:
-   `dlib`
-   `opencv-python`
-   `face_recognition`
-   `pyvirtualcam`
-   `numba`

#### For GPU Installation:
*Note: You do not need to install the `dlib` library using pip, as it must be built with CUDA support.*
-   `opencv-python`
-   `face_recognition`
-   `pyvirtualcam`
-   `numba`

---

## License

Copyright (c) 2025 Roadbobek

This software is licensed under the MIT License.

See the LICENSE.txt file for details

#### MIT License