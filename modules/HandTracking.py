import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        # init hand detection and traking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode, max_num_hands,
                                        min_detection_confidence,
                                        min_tracking_confidence)

        # drawing utilsx
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, drawLandmarks=False):
        # re-arranging frame's color order to RGB from BGR, as mediapipe uses only RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # processes RGB frame
        self.results = self.hands.process(frameRGB)

        # if a hand is detected, and drawing is true
        if self.results.multi_hand_landmarks and drawLandmarks:
            # iterating on each hand
            for handLandmarks in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLandmarks,
                                           self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNumber=0, draw=False):
        landmarks = []

        height, width, channel = frame.shape

        # if a hand is detected, and drawing is true
        if self.results.multi_hand_landmarks:
            # selecting hand
            currentHand = self.results.multi_hand_landmarks[handNumber]

            # iterating on each point of the hand. There are total 21 points on a hand. Landmark will give x, y and z axis values as a real number
            for id, landmark in enumerate(currentHand.landmark):

                # cx and cy are positions from the center point
                cx, cy = int(width * landmark.x), int(height * landmark.y)
                landmarks.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        return landmarks, self.results.multi_handedness





# testing current module
'''
This block is used to test this module
'''
if __name__ == '__main__':
    # 0, laptop's camera
    camStream = cv2.VideoCapture(0)

    wCam, hCam = 640, 480
    camStream.set(3, wCam)
    camStream.set(4, hCam)
    # variables for fps
    currentTime = 0
    previousTime = 0
    fpsPosition = (10, 80)
    fpsScale = 3
    fpsColor = (0, 255, 0)  # fpsColor in BGR
    fpsThikness = 3

    detector = HandDetector(max_num_hands=1)

    while True:
        # success will be true if reading a frame from camera was successful, img will return frame
        success, img = camStream.read()

        if success:
            # flipping the frame horizontally
            img = cv2.flip(img, 1)

            # finding hands
            img = detector.findHands(img, draw=True)

            lmList = detector.findPosition(img, draw=True)

            if lmList:
                xThumb, yThumb = lmList[4][1], lmList[4][2]
                xIndexFinger, yIndexFinger = lmList[8][1], lmList[8][2]

                cv2.circle(img, (xThumb, yThumb), 15, (255, 0, 255),
                           cv2.FILLED)
                cv2.circle(img, (xIndexFinger, yIndexFinger), 15,
                           (255, 0, 255), cv2.FILLED)
                cv2.line(img, (xThumb, yThumb), (xIndexFinger, yIndexFinger),
                         (255, 0, 255), 3)

            # calculating fps
            currentTime = time.time()
            fps = int(1 / (currentTime - previousTime))
            previousTime = currentTime

            # displaying fps text
            # cv2.putText(img, str(fps), fpsPosition, cv2.FONT_HERSHEY_PLAIN, fpsScale, fpsColor, fpsThikness)

            # displays window with frame
            cv2.imshow("Video", img)

            # wait for 1 ms to load another frame
            cv2.waitKey(1)
