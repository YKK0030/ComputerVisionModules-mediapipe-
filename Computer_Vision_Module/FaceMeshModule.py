import cv2
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, mode=False, maxFaces=2, minDetectionCon=0.5, minTrackingCon=0.5):

        self.mode = mode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces,self.minDetectionCon,self.minTrackingCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius= 2)

    def findMesh(self, img, draw=False):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACE_CONNECTIONS,
                                  self.drawSpec,self.drawSpec)
                face = []
                for id, lm in enumerate(facelms.Landmark):
                    print(lm)
                    h, w, c = img.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read
        img, faces = detector.findMesh()
        if len(faces) != 0:
            print(len(faces))
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'FPS: {int(fps)}', (28,70), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        cv2.imshow('image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()