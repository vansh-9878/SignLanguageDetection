import cv2
import handTrackModule as htm
import pandas as pd
import joblib
import numpy as np

cap=cv2.VideoCapture(0)
hand=htm.handDetector(max=1)
# V2 best for now
model=joblib.load('modelV3.pkl')


def predictSign(data):
    df=pd.DataFrame(data)
    df2=pd.DataFrame()
    # print(df.columns)
    for col in df.columns:
        # col=str(col)
        # print(col)
        first=str(col)+'x'
        second=str(col)+'y'
        df2[first]=df[col].astype(str).str.strip('[]').str.split(',').str[1].astype('int')
        df2[second]=df[col].astype(str).str.strip('[]').str.split(',').str[2].astype('int')
    
    # print(df2.columns)
    predictions=model.predict(df2)
    return predictions[0]
    # return 0



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
    
    
    