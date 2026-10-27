import cv2

# Open the laptop's default webcam
camera = cv2.VideoCapture(0)

# Check whether the camera opened successfully
if not camera.isOpened():
    print("ERROR: Could not access the webcam.")
    print("Check camera permissions or close other apps using it.")
    exit()

print("NEXUS: Webcam connected!")
print("Press Q to close the camera window.")

while True:
    # Capture one frame from the webcam
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a camera frame.")
        break

    # Mirror the image to behave like a mirror
    frame = cv2.flip(frame, 1)

    # Display the live camera feed
    cv2.imshow("NEXUS - Vision System", frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()

print("NEXUS: Vision system shut down.")