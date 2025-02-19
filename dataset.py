import cv2
import handTrackModule as htm
import pandas as pd
import os

cap=cv2.VideoCapture(0)
model=htm.handDetector(max=1)
data=[]
labels=[]
file_path="signLanguageDatset2.csv"

while True:
    ret,frame=cap.read()
    img=cv2.flip(frame,1)
    model.findHands(img,draw=False)
    lmlist=model.findPosition(img)
    # print(lmlist)
    if(len(lmlist)!=0):
        data.append(lmlist)
        labels.append(2)
    
    if len(data) == 100:
        df = pd.DataFrame(data)
        df['label'] = labels

        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)  
        else:
            df.to_csv(file_path, mode='w', header=True, index=False) 

        print("Data appended successfully!")
        break
    
    if (ret==False):
        break
    
    cv2.imshow("Video",img)
    key=cv2.waitKey(1)
    if key==27:
        break
    
cv2.destroyAllWindows()
    
    
    