import cv2
import mediapipe as mp
import os
import time
import pyautogui
pinch_was_active = False
scroll_reference_y = None
scroll_threshold = 25
scroll_speed = 2
screen_width, screen_height = pyautogui.size()


# ==========================================
# NEXUS: HAND TRACKING SYSTEM
# Version 1.0
# ==========================================

print("NEXUS: Initialising vision system...")

# Locate the hand-tracking model
project_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    project_folder, "models", "hand_landmarker.task"
)

# Check that the model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Hand model not found: {model_path}\n"
        "Check that hand_landmarker.task is inside the models folder."
    )

# Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Connections between hand landmarks
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def recognise_gesture(points, width):
    # Check whether the four fingers are extended.
    index_up = points[8][1] < points[6][1]
    middle_up = points[12][1] < points[10][1]
    ring_up = points[16][1] < points[14][1]
    pinky_up = points[20][1] < points[18][1]

    fingers_up = sum([
        index_up,
        middle_up,
        ring_up,
        pinky_up
    ])

    # Measure the distance between the thumb and index fingertip.
    thumb_x, thumb_y = points[4]
    index_x, index_y = points[8]

    distance = (
        (thumb_x - index_x) ** 2 +
        (thumb_y - index_y) ** 2
    ) ** 0.5

    # Recognise the gesture.
    if distance < width * 0.06:
        return "PINCH"

    elif fingers_up == 4:
        return "OPEN HAND"
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "SCROLL"

    elif fingers_up == 0:
        return "FIST"

    else:
        return "UNKNOWN"
# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "NEXUS could not access the webcam. "
        "Close other camera applications and try again."
    )

print("NEXUS: Webcam connected!")
print("NEXUS: Hand tracking activated.")
print("Move your hand in front of the camera.")
print("Press Q to exit.")

frame_number = 0

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("NEXUS: Could not read webcam frame.")
                break

            # Mirror the image for natural movement
            frame = cv2.flip(frame, 1)

            # Convert OpenCV's BGR image to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Create a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Process the frame
            frame_number += 1
            timestamp_ms = frame_number * 33

            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # Draw detected hands
            if results.hand_landmarks:

                for hand in results.hand_landmarks:

                    height, width, _ = frame.shape
                    points = []
                    

                    # Convert landmark coordinates to pixels
                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))
                    gesture = recognise_gesture(points, width)
                    # Use the index fingertip to control the cursor
                    index_x, index_y = points[8]
                    # Map webcam coordinates to screen coordinates.
                    cursor_x = int(index_x / width * (screen_width - 1))
                    cursor_y = int(index_y / height * (screen_height - 1))
                    # Move the cursor while hand is open.
                    if gesture == "OPEN HAND":
                        pyautogui.moveTo(cursor_x, cursor_y)
                    elif gesture == "SCROLL":
                                     if scroll_reference_y is None:
                                      scroll_reference_y = index_y
                                     else:
                                           movement = scroll_reference_y - index_y

                                           if abs(movement) >= scroll_threshold:
                                                 scroll_amount = int(movement * scroll_speed / 5)

                                                 if scroll_amount != 0:
                                                             pyautogui.scroll(scroll_amount)

                                                 scroll_reference_y = index_y
                    #click once for each pinch
                    if gesture == "PINCH":
                        if not pinch_was_active:
                            pyautogui.click()
                            pinch_was_active = True
                    else:
                        pinch_was_active = False
                    
                    cv2.putText(frame, f"Gesture: {gesture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                        

                    # Draw connections between finger joints
                    for start, end in CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    # Draw the individual landmarks
                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                hand_count = len(results.hand_landmarks)

                cv2.putText(
                    frame,
                    f"NEXUS: {hand_count} hand(s) detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "NEXUS: Searching for hands...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )

            # Display the result
            cv2.imshow("NEXUS - Hand Tracking", frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("NEXUS: Vision system shut down.")