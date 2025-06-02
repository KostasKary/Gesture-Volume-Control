import cv2
import mediapipe as np
import time
import math

class handDetect():

    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxhands=maxHands
        self.modelComplex=modelComplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=np.solutions.hands
        self.hands=self.mpHands.Hands(self.mode, self.maxhands, self.modelComplex, self.detectionCon, self.trackCon)
        self.npDraw=np.solutions.drawing_utils
        self.tipIds=[4, 8, 12, 16, 20]

    def findHand(self, img, draw=True):

        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.npDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist=[]

        if self.results.multi_hand_landmarks:

            myhand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])

                if draw:
                    cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

                Xmin,Xmax=min(xList),max(xList)
                Ymin,Ymax=min(yList),max(yList)
                bbox=Xmin,Xmax,Ymin,Ymax

        return self.lmlist, bbox

    def fingersup(self):
        finger=[]
        if self.lmlist!=[]:
            if self.lmlist[self.tipIds[0]] [1]<self.lmlist[self.tipIds[0]-1][1]:
                finger.append(0)
            else:
                finger.append(1)

            for id in range(1,5):
                if self.lmlist[self.tipIds[id]] [2]<self.lmlist[self.tipIds[id]-2][2]:
                    finger.append(1)

                else:
                    finger.append(0)

            return finger

    def Distance(self,img,Top_1,Top_2,draw=True):
        x1,y1=self.lmlist[Top_1][1:]
        x2,y2=self.lmlist[Top_2][1:]

        cx,cy=(x1+x2)//2,(y1+y2)//2

        length=math.hypot(x1-x2,y1-y2)

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(img,(cx,cy),7,(0,0,255),cv2.FILLED)
        return length, img, [x1, x2, y1, y2, cx, cy]