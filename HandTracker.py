import time
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self,mode = False,maxHands = 2,
                 modelComplexity = 1,
                 detectionConfidence = 0.5,
                 trackConfidence = 0.5
                 ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.modelComplexity,
            self.detectionConfidence,
            self.trackConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.landMarkList = None
        self.middleTip = 12
        self.indexTip = 8
        self.thumbTip = 4
        self.ringTip = 16
        self.pinkyTip = 20
        self.fingerTips = [
            self.thumbTip ,
            self.indexTip ,
            self.middleTip,
            self.ringTip,
            self.pinkyTip
        ]


    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw : 
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)


        return img

    def findPosition(self,img,handNo= 0,draw = True): 
        lmList = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNo]

            for id,landmarks in enumerate(handLms.landmark):
                #print(id,landmarks)
                h, w, c = img.shape
                cx, cy = int(landmarks.x*w), int(landmarks.y*h)
                lmList.append([id,cx,cy])
                ## For index finger tracking
                if id == 8 and draw:
                    cv2.circle(img,(cx,cy), 25, (255,0,255),cv2.FILLED)
        self.landMarkList = lmList
        return lmList
    
    def fingersUp(self,img):
        fingersUp = 0
        if len(self.landMarkList) > 0 :
            if self.landMarkList[self.thumbTip][1] < self.landMarkList[self.thumbTip-2][1]:
                fingersUp += 1
            for i in range(1,5):
                if self.landMarkList[self.fingerTips[i]][2] < self.landMarkList[self.fingerTips[i]-2][2]:
                    fingersUp+=1
        return fingersUp

    def findIfStart(self,img):
        if len(self.landMarkList) > 0:
            if self.landMarkList[self.indexTip][2] < self.landMarkList[self.indexTip-2][2] and self.landMarkList[self.middleTip][2] < self.landMarkList[self.middleTip-2][2] and self.fingersUp(img) == 2:
                return True
            else:
                return False

    def findIfDraw(self,img):
        if len(self.landMarkList) > 0:
            if self.landMarkList[self.indexTip][2] < self.landMarkList[self.indexTip-2][2] and self.fingersUp(img) == 1 :
                return True
            else:
                return False
        
        


        


# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)   
#     detector = handDetector()
#     while True:
        
#         success, img = cap.read()
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) > 0:
#             print(lmList[8])
#         cTime = time.time()
#         fps = 1/(cTime-pTime)
#         pTime = cTime
        
#         cv2.putText(img,"FPS : " + str(fps),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

#         cv2.imshow("Image",img)
#         keyPressed = cv2.waitKey(1)
#         if keyPressed != -1:
#             break

# if __name__ == '__main__':
#     main()