import cv2
import handTrackModule as htm
import pandas as pd
import joblib
import numpy as np

cap=cv2.VideoCapture(0)
hand=htm.handDetector(max=1)
# V2 best for now
model=joblib.load('modelV4.pkl')


def predictSign(data):
    # data is a list containing the landmarks, where each landmark is [id, x, y]
    # We'll assume data[0] is the list of landmarks.
    landmarks = pd.DataFrame(data[0], columns=["id", "x", "y"])
    
    # Create a new DataFrame to hold our computed distance features.
    df2 = pd.DataFrame()
    
    # Compute the pairwise Euclidean distances between landmarks.
    # We loop over unique pairs (i, j) where i < j.
    for i in range(len(landmarks)):
        for j in range(i+1, len(landmarks)):
            x1 = landmarks.loc[i, "x"]
            y1 = landmarks.loc[i, "y"]
            x2 = landmarks.loc[j, "x"]
            y2 = landmarks.loc[j, "y"]
            
            # Calculate Euclidean distance.
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # Store the distance as a feature. We wrap dist in a list to create a single-row DataFrame.
            df2[f"dist_{i}_{j}"] = [dist]
    
    # Use the model to predict based on these features.
    predictions = model.predict(df2)
    return predictions[0]


while True:
    ret,frame=cap.read()
    img=cv2.flip(frame,1)
    hand.findHands(img,draw=False)
    lmlist=hand.findPosition(img)
    # print(lmlist)
    if(len(lmlist)!=0):
        data=[]
        data.append(lmlist)
        sign=predictSign(data)
        
        index=np.where(sign==1)[0]
        if index.size >0:
            ans=index[0]   
            cv2.putText(img,"Guess : "+chr(ans+65),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        
    if (ret==False):
        break
    
    cv2.imshow("Video",img)
    key=cv2.waitKey(1)
    if key==27:
        break
    
cv2.destroyAllWindows()
    
    
    