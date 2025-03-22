import cv2
import handTrackModule as htm
import numpy as np

cap=cv2.VideoCapture(0)
hand=htm.handDetector(max=1)

while True:
    ret,frame=cap.read()
    img=cv2.flip(frame,1)
    hand.findHands(img,draw=False)
    lmlist=hand.findPosition(img,draw=True)
    # print(lmlist)
    if(len(lmlist)!=0):
        landmarks = np.array([(x, y) for _, x, y in lmlist])

        x_min, y_min = np.min(landmarks, axis=0)  
        x_max, y_max = np.max(landmarks, axis=0)  

        padding = 20
        x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
        x_max, y_max = min(frame.shape[1], x_max + padding), min(frame.shape[0], y_max + padding)

        cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,0,255),2)
        cropped_hand = img[y_min:y_max, x_min:x_max]
        cropped_hand=cv2.resize(cropped_hand,(255,255))
        cv2.imshow("Crop",cropped_hand)
    
    
    if (ret==False):
        break
    
    cv2.imshow("Video",img)
    key=cv2.waitKey(1)
    if key==27:
        break
    
cv2.destroyAllWindows()
    
    
    