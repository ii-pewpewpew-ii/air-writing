import mediapipe as mp
import HandTracker as ht
import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = ht.handDetector(detectionConfidence=0.85)
xp,yp =0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img,draw=False)

    if len(landMarkList) != 0:
        x1,y1 = landMarkList[8][1:]
        x2,y2 = landMarkList[12][1:]
    if detector.findIfStart(img):
        xp, yp = 0, 0
        print("Starting Track")
        cv2.rectangle(img,(x1,y1-15),(x2,y2+15),(0,0,200),cv2.FILLED)
    if detector.findIfDraw(img):
        cv2.circle(img,(x1,y1),5,(100,200,65))
        print("Drawing Now")
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
    
        cv2.line(img,(xp,yp),(x1,y1),(100,200,65),5)
        cv2.line(imgCanvas,(xp,yp),(x1,y1),(100,200,65),5)
        xp,yp = x1,y1
    
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,  imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break
