import cv2
import mediapipe as mp
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np


class Player:
    def __init__(self):
        self.hand = []  # This initializes the hand attribute as an empty list.

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe Hands and Drawing Utilities
        self.hands = mp.solutions.hands.Hands(static_image_mode=self.mode, 
                                              max_num_hands=self.maxHands, 
                                              min_detection_confidence=self.detectionCon, 
                                              min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utilities
        self.tipIds = [4, 8, 12, 16, 20]  # Landmark IDs for fingertips

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList, bbox = [], [], []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return fingers  # Return an empty list if no landmarks are found
        
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]: 
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if len(self.lmList) == 0:
            return 0, img, (0, 0, 0, 0, 0, 0)  # Return zero distance if no landmarks
        
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2] 
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2] 
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, (x1, y1, x2, y2, cx, cy)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Open the default camera
    detector = handDetector()

    # Initialize Pycaw for volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0

    while True:
        success, img = cap.read()  # Capture frame-by-frame
        if not success:
            print("Failed to capture image")
            break

        img = detector.findHands(img)  # Detect hands
        lmList, bbox = detector.findPosition(img)  # Get positions and bounding box

        if len(lmList) != 0:
            # Calculate the distance between the thumb and index finger
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # Convert length to volume level
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            # Set volume level
            volume.SetMasterVolumeLevel(vol, None)

            # Draw a volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Image", img)  # Display the resulting frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break

    cap.release()  # Release the capture
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
