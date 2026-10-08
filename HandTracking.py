import cv2
import sys
import time
import mediapipe as mp
import matplotlib.pyplot as 

# -_-_-_-_-_-_-_-_-_-___-_

    #Include angle calculations, since defining angles and getting angle values may be easer to read and deal with rather than just x and y cords.







# ── Tasks API (replaces mp.solutions.hands which no longer exists) ─────────
# Bseoptions is used to specify the model file
BaseOptions           = mp.tasks.BaseOptions
# HandLandmarker is the main class for hand tracking in mediapipe tasks 
HandLandmarker        = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions is used to specify options for the hand landmarker, such as the model file and running mode.
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode is used to specify the running mode for the hand landmarker, such as video or image mode.
VisionRunningMode     = mp.tasks.vision.RunningMode

# Same connections as the old mp_hands.HAND_CONNECTIONS
HAND_CONNECTIONS = [
    # (start, end) pairs for the 21 hand landmarks
    # (0,1) means a connection between landmark 0 and landmark 1, and so on
            #landmark 0 is the wrist, 1-4 are the thumb, 5-8 are the index finger, 9-12 are the middle finger, 13-16 are the ring finger, and 17-20 are the pinky finger
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

GESTURE_COOLDOWN_MS = 2000

def mp_draw_landmarks(frame, landmarks):
    """Replaces mp_draw.draw_landmarks()"""
    # Get the height and width of the frame to convert normalized coordinates to pixel coordinates
    h, w = frame.shape[:2]
    for start, end in HAND_CONNECTIONS:
        # Convert normalized coordinates to pixel coordinates
        x0, y0 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x1, y1 = int(landmarks[end].x * w),   int(landmarks[end].y * h)
        # Draw the connection line
        cv2.line(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
        # for lm in [landmarks[start], landmarks[end]]:
    for lm in landmarks:
        # Draw the landmark point
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)


def debounced_print(message, now_ms, state, cooldown_ms=GESTURE_COOLDOWN_MS):
    if message == state["last_message"]:
        return
    if now_ms - state["last_print_ms"] < cooldown_ms:
        return

    print(message)
    state["last_message"] = message
    state["last_print_ms"] = now_ms


#_-_-_-_-_-_-_-_-_-_-_-MediaPIPE section-_-_-_-_-_-_-_-_-_-_-_-#



## this uses openCV to pull video from the webcam,
#  and then uses the mediapipe tasks API to track the hand landmarks in the video stream,
#  and displays the video stream with the hand landmarks drawn on it in a window
def track_hands():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )

    stream = cv2.VideoCapture(0)
    windowName = "Hand Tracker"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # The Tasks API requires a timestamp for each video frame, so we record the start time and calculate elapsed time for each frame.
    start_time = time.time()
    gesture_state = {"last_message": None, "last_print_ms": 0}

    with HandLandmarker.create_from_options(options) as landmarker:
        while cv2.waitKey(1) != 27:
            has_frame, frame = stream.read()
            if not has_frame:
                print("Unable to capture video")
                break

            # Convert and detect (replaces hands.process(RGB))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # change the input to a MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Calculate the timestamp in milliseconds for the current frame
            timestamp_ms = int((time.time() - start_time) * 1000)
            #
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # draws the hand landmarks (knuckles and tips) and connections between them, and checks which fingers are up
            if results.hand_landmarks:
                # create a list of the hand landmarks for each hand detected in the frame
                for handlms in results.hand_landmarks:
                    # place the hand landmark on the frame (screen) using the mp_draw_landmarks function defined above
                    mp_draw_landmarks(frame, handlms)  # replaces mp_draw.draw_landmarks()

                    index_tip    = handlms[8]
                    index_knuckle = handlms[5]
                    middle_tip   = handlms[12]
                    middle_knuckle = handlms[9]
                    ring_tip     = handlms[16]
                    ring_knuckle = handlms[13]
                    pinky_tip    = handlms[20]
                    pinky_knuckle = handlms[17]

                    if index_tip.y < index_knuckle.y:
                        debounced_print("Index finger is up", timestamp_ms, gesture_state)
                    if middle_tip.y < middle_knuckle.y:
                        debounced_print("Middle finger is up", timestamp_ms, gesture_state)
                    if ring_tip.y < ring_knuckle.y:
                        debounced_print("Ring finger is up", timestamp_ms, gesture_state)
                    if pinky_tip.y < pinky_knuckle.y:
                        debounced_print("Pinky finger is up", timestamp_ms, gesture_state)
                    if (index_tip.y < index_knuckle.y and middle_tip.y < middle_knuckle.y
                            and ring_tip.y < ring_knuckle.y and pinky_tip.y < pinky_knuckle.y):
                        debounced_print("All fingers are up", timestamp_ms, gesture_state)

            cv2.imshow(windowName, frame)

    stream.release()
    cv2.destroyAllWindows()



# this function will find certain shapes made by hands, such as a fist, open hand, or peace sign, by analyzing the positions of the hand landmarks 
        # this is to get the basic shape recogntion working then be utilized in different methods
def find_hand_shapes():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )

    stream = cv2.VideoCapture(0)
    windowName = "Hand Tracker"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # The Tasks API requires a timestamp for each video frame, so we record the start time and calculate elapsed time for each frame.
    start_time = time.time()
    gesture_state = {"last_message": None, "last_print_ms": 0}

    with HandLandmarker.create_from_options(options) as landmarker:
        while cv2.waitKey(1) != 27:
            has_frame, frame = stream.read()
            if not has_frame:
                print("Unable to capture video")
                break

            # Convert and detect (replaces hands.process(RGB))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # change the input to a MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Calculate the timestamp in milliseconds for the current frame
            timestamp_ms = int((time.time() - start_time) * 1000)
            #
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # draws the hand landmarks (knuckles and tips) and connections between them, and checks which fingers are up
            if results.hand_landmarks:
                # create a list of the hand landmarks for each hand detected in the frame
                for handlms in results.hand_landmarks:
                    # place the hand landmark on the frame (screen) using the mp_draw_landmarks function defined above
                    mp_draw_landmarks(frame, handlms)  # replaces mp_draw.draw_landmarks()

                    # 0 - writst
                    #1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
                    
                    wrist       = handlms[0]
                    thumb_tip    = handlms[4]
                    thumb_knuckle = handlms[3]
                    thumb_base    = handlms[2]
                    index_tip    = handlms[8]
                    index_knuckle = handlms[5]
                    middle_tip   = handlms[12]
                    middle_knuckle = handlms[9]
                    ring_tip     = handlms[16]
                    ring_knuckle = handlms[13]
                    pinky_tip    = handlms[20]
                    pinky_knuckle = handlms[17]

                    is_thumb_up = thumb_tip.y < thumb_base.y
                    # Thuumb extended means that the thumb is not just up, but also extended away from the hand, which can be determined by checking if the x coordinate of the thumb tip
                    #  is significantly different from the x coordinate of the index knuckle (Landmark 5)
                    thumb_extended = abs(thumb_tip.x - index_knuckle.x)

                    #hand ceiling 
                    hand_ceiling = min(index_knuckle.y, middle_knuckle.y)

                    #index mapping 
                    index_mag = ((index_tip.x - index_knuckle.x)**2 + (index_tip.y - index_knuckle.y)**2)**0.5
                    index_extended = index_mag > 0.08

                    #middle mapping 
                    middle_mag = ((middle_tip.x - middle_knuckle.x)**2 + (middle_tip.y - middle_knuckle.y)**2)**0.5
                    middle_extended = middle_mag > 0.08

                    #ring mapping 
                    ring_mag = ((ring_tip.x - ring_knuckle.x)**2 + (ring_tip.y - ring_knuckle.y)**2)**0.5
                    ring_extended = ring_mag > 0.08

                    #pinky mapping 
                    pinky_mag = ((pinky_tip.x - pinky_knuckle.x)**2 + (pinky_tip.y - pinky_knuckle.y)**2)**0.5
                    pinky_extended = pinky_mag > 0.08

                    fingers_curled = (index_tip.y)

                    #camera is flipped so x values are reversed, so we check if the tip is to the left of the knuckle to determine if the finger is up or down

                    if thumb_extended > 0.15:
                        # check the other fingers are curled in
                        if (index_tip.y <= index_knuckle.y and middle_tip.y <= middle_knuckle.y 
                            and ring_tip.y <= ring_knuckle.y and pinky_tip.y <= pinky_knuckle.y):
                                # check if the thumb is pointing upward by checking its y cord
                                # is the largest y value of all landmarks on screen  
                                if (thumb_tip.y > thumb_knuckle):
                                    debounced_print("Thumbs Up", timestamp_ms, gesture_state)

                    # check for peace sign, which is index and middle up, ring and pinky down
                    if (index_tip.y < index_knuckle.y and middle_tip.y < middle_knuckle.y
                        and ring_tip.y > ring_knuckle.y and pinky_tip.y > pinky_knuckle.y):
                        debounced_print("Peace Sign", timestamp_ms, gesture_state)

                    # check for rock and roll sign, which is index and pinky up, middle and ring down
                    if (index_tip.y > index_knuckle.y and pinky_tip.y > pinky_knuckle.y
                        and middle_tip.y <= middle_knuckle.y and ring_tip.y <= ring_knuckle.y):
                        debounced_print("Rock and Roll", timestamp_ms, gesture_state)

                    is_horizontal = abs(index_tip.x - index_knuckle.x) > abs(index_tip.y - index_knuckle.y)

                    # index finger pointing left or right 
                    if index_extended and is_horizontal:
                        if index_tip.x < index_knuckle.x:
                            debounced_print("index finger is pointing left in frame (right in real life)", timestamp_ms, gesture_state)
                        else:
                            debounced_print("index finger is pointing right in frame (left in real life)", timestamp_ms, gesture_state)

                    #index finger pointing up or down
                    if index_extended and not is_horizontal:
                        if index_tip.y < index_knuckle.y:
                            debounced_print("index finger is pointing up in frame", timestamp_ms, gesture_state)
                        else:
                            debounced_print("index finger is pointing down in frame", timestamp_ms, gesture_state)

                    #middle finger pointing up or down
                    if middle_extended and not is_horizontal:
                        if middle_tip.y < middle_knuckle.y:
                            debounced_print("middle finger is pointing up in frame", timestamp_ms, gesture_state)
                        else:
                            debounced_print("middle finger is pointing down in frame", timestamp_ms, gesture_state)

                    #ring finger pointing up or down
                    if ring_extended and not is_horizontal:
                        if ring_tip.y < ring_knuckle.y:
                            debounced_print("ring finger is pointing up in frame", timestamp_ms, gesture_state)
                        else:
                            debounced_print("ring finger is pointing down in frame", timestamp_ms, gesture_state)

                    #pinky finger pointing up or down
                    if pinky_extended and not is_horizontal:
                        if pinky_tip.y < pinky_knuckle.y:
                            debounced_print("pinky finger is pointing up in frame", timestamp_ms, gesture_state)
                        else:
                            debounced_print("pinky finger is pointing down in frame", timestamp_ms, gesture_state)

                    # check for open handing pointing left
                    if (index_tip.x < index_knuckle.x and middle_tip.x < middle_knuckle.x
                            and ring_tip.x < ring_knuckle.x and pinky_tip.x < pinky_knuckle.x):
                        debounced_print("Open hand pointing left (right in real life)", timestamp_ms, gesture_state)

                     # check for open hand pointing right 
                    if (index_tip.x > index_knuckle.x and middle_tip.x > middle_knuckle.x
                            and ring_tip.x > ring_knuckle.x and pinky_tip.x > pinky_knuckle.x):
                        debounced_print("Open hand pointing right (left in real life)", timestamp_ms, gesture_state)
            cv2.imshow(windowName, frame)
    stream.release()
    cv2.destroyAllWindows()

                    

                    



if __name__ == "__main__":
    find_hand_shapes()