import cv2
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "TopBar"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255)
brushThickenss = 15
eraserThickenss = 50

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

##
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
##
detector = htm.handDetector(detectionCon=0.65)
xp, yp = 0, 0
while True:
    # image import
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # find HAnd Landmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        fingers = detector.fingersUp()
        print(fingers)

        # if selection mode (2 fing up)
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            xp, yp = x1, y1
            print("selection")
            # check for click
            if y1 < 125:
                if 228 < x1 < 402:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 531 < x1 < 725:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 820 < x1 < 1010:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 1094 < x1 < 1269:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)


        # if drawing mode (index)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 20, drawColor, cv2.FILLED)
            print("Pointing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickenss)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickenss)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickenss)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickenss)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # header
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("AeroPainter", img)
    #cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
