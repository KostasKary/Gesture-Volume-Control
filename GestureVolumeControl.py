import cv2
import time
import HandTrackingModule as htm

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import math

length=0

wCam, hCam=640,480

cap=cv2.VideoCapture(0)

detector=htm.handDetect(detectionCon=0.7)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume=cast(interface, POINTER(IAudioEndpointVolume))

volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]

vol=0
volBar=400
volPer=0

cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

while True:
    success, img=cap.read()

    img=detector.findHand(img)
    lmlist=detector.findPosition(img)
    print(lmlist)
    if lmlist!=([],[]):
        length, img, [x1, x2, y1, y2, cx, cy] = detector.Distance(img, 4, 8, draw=True)

    vol=np.interp(length, [50,300], [minVol, maxVol])
    volBar=np.interp(length, [50,300], [400, 150])
    volPer=np.interp(length,[50,300], [0, 100])

    volume.SetMasterVolumeLevel(vol, None)
    print("length: "+ str(length)+", vol: "+ str(vol))

    cv2.rectangle(img,(0,200),(50,300),(255,0,0),3)
    cv2.rectangle(img,(0,int(volBar)),(50,300),(255,0,0), cv2.FILLED)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, "FPS"+str(int(fps)),(40,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cv2.imshow("Gesture Volume Control", img)

    k=cv2.waitKey(1)
    if k%256==27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
