import cv2
import handTrackModule as htm
import pandas as pd
import joblib


cap=cv2.VideoCapture(0)
hand=htm.handDetector(max=1)
model=joblib.load('modelV1.pkl')


def predictSign(data):
    df=pd.DataFrame(data)
    df2=pd.DataFrame()
    for col in df.columns:
        col=str(col)
        first=col+'x'
        second=col+'y'
        df2[first]=df[col].astype(str).str.strip('[]').str.split(',').str[1].astype('int')
        df2[second]=df[col].astype(str).str.strip('[]').str.split(',').str[2].astype('int')
    
    predictions=model.predict(df2)
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
        print(sign)    
        
    if (ret==False):
        break
    
    cv2.imshow("Video",img)
    key=cv2.waitKey(1)
    if key==27:
        break
    
cv2.destroyAllWindows()
    
    
    