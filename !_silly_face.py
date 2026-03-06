import os

# --- CUDA Paths ---
USE_GPU = True
try:
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64")
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v9.12/bin/13.0")
except:
    USE_GPU = False

import cv2
import face_recognition
import random
import time

import cv2
import face_recognition
import random
import time

# --- Settings ---
FACE_COLOUR = (0, 255, 0)  # Green (BGR)
BG_BOX_COLOUR = (0, 0, 0)  # Black box
BUNNY_EARS = "(\_/)"
BUNNY_EARS_TOGGLE = True

class FaceCompanion:
    def __init__(self):
        self.mood = "Neutral"
        self.blink_state = False
        self.last_mood_change = time.time()
        self.last_blink = time.time()
        self.blink_duration = 0.15
        self.next_blink_gap = random.uniform(2, 4)

        self.faces = {
            "Neutral": "(o_o)", "Happy": "(^V^)", "Sad": "(;_;)",
            "Angry": "(>_<)", "Bored": "(~_~)", "Confused": "(o.o)",
            "Shocked": "(0.0)", "Tweaked": "(o.0)", "Pleased": "(ouo)",
            "Gay": "(uwu)", "More_Gay": "(owo)", "Tired": "(=_=)",
            "Excited": "(*o*)", "Stare": "(o o)"
        }
        self.faces_blink = {
            "Neutral": "(-_-)", "Happy": "(-v-)", "Sad": "(-_-)",
            "Angry": "(-_-)", "Bored": "(-_-)", "Confused": "(-.-)",
            "Shocked": "(-.-)", "Tweaked": "(o.-)", "Pleased": "(-u-)",
            "Gay": "(-w-)", "More_Gay": "(-w-)", "Tired": "(-,-)",
            "Excited": "(-o-)", "Stare": "(- -)"
        }

    def update(self):
        now = time.time()
        if now - self.last_mood_change > 5:
            self.mood = random.choice(list(self.faces.keys()))
            self.last_mood_change = now
        if not self.blink_state and (now - self.last_blink > self.next_blink_gap):
            self.blink_state = True
            self.last_blink = now
        elif self.blink_state and (now - self.last_blink > self.blink_duration):
            self.blink_state = False
            self.last_blink = now
            self.next_blink_gap = random.uniform(2, 4)

    def get_face(self):
        return self.faces_blink[self.mood] if self.blink_state else self.faces[self.mood]

# --- Init ---
video_capture = cv2.VideoCapture(0)
companion = FaceCompanion()
model_type = "cnn" if USE_GPU else "hog"

while True:
    ret, frame = video_capture.read()
    if not ret: break

    companion.update()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame, model=model_type)

    if len(face_locations) > 0:
        for (top, right, bottom, left) in face_locations:
            # Scale back to original size
            top *= 4; right *= 4; bottom *= 4; left *= 4

            # --- 1. Calculate 1x1 Square Box ---
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            # Find the largest dimension to keep it square
            size = int(max(right - left, bottom - top) * 1.7)

            s_left = max(0, center_x - size // 2)
            s_top = max(0, center_y - size // 2)
            s_right = min(frame.shape[1], center_x + size // 2)
            s_bottom = min(frame.shape[0], center_y + size // 2)

            # Draw the 1x1 Solid Black Square
            cv2.rectangle(frame, (s_left, s_top), (s_right, s_bottom), BG_BOX_COLOUR, -1)

            # --- 2. Text Logic ---
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = (s_right - s_left) / 200.0 # Scales with the square size
            thickness = max(1, int(font_scale * 2))

            # Draw Ears (Centered)
            if BUNNY_EARS_TOGGLE:
                (e_w, e_h), _ = cv2.getTextSize(BUNNY_EARS, font, font_scale, thickness)
                cv2.putText(frame, BUNNY_EARS, (center_x - e_w // 2, center_y - int(e_h * 0.5)),
                            font, font_scale, FACE_COLOUR, thickness)

            # Draw Face (Centered)
            face_text = companion.get_face()
            (f_w, f_h), _ = cv2.getTextSize(face_text, font, font_scale, thickness)
            cv2.putText(frame, face_text, (center_x - f_w // 2, center_y + int(f_h * 1.2)),
                        font, font_scale, FACE_COLOUR, thickness)
    else:
        # --- 3. Cut feed if no face is found ---
        frame[:] = BG_BOX_COLOUR # Fill whole frame with black
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "0.o", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                    font, 5, (255, 255, 255), 10)

    cv2.imshow('Companion Overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
