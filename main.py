import random

import cv2

import modules.HandTracking as ht
import helpers.helper as hp

camStream = cv2.VideoCapture(0)
detector = ht.HandDetector(max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

while True:
    success, frame = camStream.read()
    if success:

        # flipping the frame horizontally
        frame = cv2.flip(frame, 1)

        # # finding hands
        frame = detector.findHands(frame, drawLandmarks=True)

        # retriving landmarks values
        landmarks, handedness = detector.findPosition(frame)

        # if hand is detected in the frame then `landmarks` will have some value
        if landmarks:
            gesture, appropriate = hp.getGesture(landmarks, handedness)

            # selects gesture label randomly where there are same gesture having different lables
            # if "/" in gesture:
            #    gesture = random.choice(gesture.split("/"))


            # blur the frame if gesture is not appropriate
            if not appropriate:
               quality = 0.05  # 5 %
               height, width = frame.shape[:2]
               w, h = int(width * quality), int(height * quality)
               #  w, h = 10, 10
               #  height, width = frame.shape[:2]
               tempFrame = cv2.resize(frame, (w, h),
                                       interpolation=cv2.INTER_LINEAR)
               frame = cv2.resize(tempFrame, (width, height),
                                   interpolation=cv2.INTER_NEAREST)

            hp.dispalyLabel(frame, gesture)

        # displays the frame
        cv2.imshow("Gesture Recognition", frame)

        # two frames will have cv.waitKey( ms ) break
        key = cv2.waitKey(1)

        # press 'q' or 'esc' to quit
        if (key == ord('q') or key == 27):
            break

camStream.release()
cv2.destroyAllWindows()
