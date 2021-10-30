import cv2
import mediapipe as mp

LABEL_LEFT = "Left"
LABEL_RIGHT = "Right"

GESTURE_FIVE = "Five/5/Stop/Paper"
GESTURE_FOUR = "Four/4"
GESTURE_INAPPROPRIATE = "Inappropriate"
GESTURE_YOLO = "Yolo"
GESTURE_PKR = "Pakistani Roadies"
GESTURE_THREE = "Three/3"
GESTURE_TWO = "Two/2/Win/Victory/Scissors"
GESTURE_ROCKON = "Rock On!"
GESTURE_THUMBSUP = "Thumbs Up/Okay!"
GESTURE_ONE = "One/1"
GESTURE_FIST = "Zero/0/Fist/Stone"




def dispalyLabel(frame,
                 text,
                 position=None,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 scale=1,
                 textColor=(0, 0xff, 0),
                 thickness=2,
                 bgColor=(0, 0, 0)):
    height, width, channel = frame.shape

    padding = 5
    percentage = 1.05

    (label_width,
     label_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    if position == None:
        textStartX = 30 + padding
        textStartY = height + padding - int(label_height * percentage)
        position = (textStartX, textStartY)

    bgThickness = cv2.FILLED

    bgPositionStart = (textStartX - padding,
                       textStartY - padding - label_height)
    bgPositionEnd = (textStartX + int(label_width * percentage) + padding,
                     int(textStartY * percentage) + padding - label_height)

    # displaying black bg of text
    cv2.rectangle(frame, bgPositionStart, bgPositionEnd, bgColor, bgThickness)

    # displaying ack message
    cv2.putText(frame, text, position, font, scale, textColor, thickness)


def getHand(handedness):
    return handedness[0].classification[0].label


def getGesture(landmarks, handedness):
    # hl = mp.solutions.hands.HandLandmark(0)

    # getting fingers array
    fingers = []
    for id in range(4, 21, 4):
        # thumb
        if id < 5:
            if getHand(handedness) == LABEL_LEFT:
                if landmarks[id][1] > landmarks[id - 2][1]:
                    fingers.append(True)
                else:
                    fingers.append(False)

            elif getHand(handedness) == LABEL_RIGHT:
                # if the hand is right then check if x axis of point 4 is greater than point 2
                if landmarks[id][1] < landmarks[id - 2][1]:
                    fingers.append(True)
                else:
                    fingers.append(False)

        # fingers
        else:
            if landmarks[id][2] < landmarks[id - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)

    # no of fingers that are up
    upFingers = fingers.count(True)



    # Gesture classification: One, two/win/victory, three, four, five/stop
    gesture = "Can't recognise"
    isAppropriate = True

    if upFingers == 5:
        # gesture = "5/Stop"
        gesture = GESTURE_FIVE

    elif upFingers == 4:
        if not fingers[0]:
            # gesture = "4"
            gesture = GESTURE_FOUR

        elif not fingers[2]:
            gesture = GESTURE_INAPPROPRIATE
            isAppropriate = False

    elif upFingers == 3:
        if fingers[0] and fingers[4]:
            if fingers[1]:
                # gesture = "YOLO!"
                gesture = GESTURE_YOLO

            elif fingers[3]:
                # gesture = "Pakistani Roadies Symbol"
                gesture = GESTURE_PKR

        else:
            # gesture = "3"
            gesture = GESTURE_THREE

    elif upFingers == 2:
        if fingers[1] and fingers[2]:
            # gesture = "Two/2/Win/Victory/Scissors"
            gesture = GESTURE_TWO

        elif fingers[1] and fingers[4]:
            # gesture = "Rock On!"
            gesture = GESTURE_ROCKON

    elif upFingers == 1:
        if fingers[0]:
            # gesture = "Thumbs Up!/Okay!"
            gesture = GESTURE_THUMBSUP

        elif fingers[1]:
            # gesture = "One"
            gesture = GESTURE_ONE

        # middle finger
        elif fingers[2] or fingers[3]:
            gesture = GESTURE_INAPPROPRIATE
            # gesture = "Inappropriate"
            isAppropriate = False

    elif upFingers == 0:
        # gesture = "Fist/Stone"
        gesture = GESTURE_FIST

    return gesture, isAppropriate
