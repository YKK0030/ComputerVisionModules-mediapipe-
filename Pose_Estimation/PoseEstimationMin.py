import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img, True)
    if len(lmList) !=0:
        print(lmList[1])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    ptime = cTime

    cv2.putText(img,str(int(fps)), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)