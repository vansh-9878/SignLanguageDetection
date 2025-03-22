import cv2
import handTrackModule as htm
import pandas as pd
import joblib
import numpy as np

cap=cv2.VideoCapture(0)
hand=htm.handDetector(max=1)
# V2 best for now
model=joblib.load('modelV1.pkl')


def calculate_angles_vectorized(A, B, C):
    """Efficiently compute angles at point B given three sets of points A, B, and C."""
    AB_x = A[:, 0] - B[:, 0]
    AB_y = A[:, 1] - B[:, 1]
    BC_x = C[:, 0] - B[:, 0]
    BC_y = C[:, 1] - B[:, 1]

    dot_product = AB_x * BC_x + AB_y * BC_y
    magnitude_AB = np.sqrt(AB_x**2 + AB_y**2)
    magnitude_BC = np.sqrt(BC_x**2 + BC_y**2)

    cosine_theta = np.clip(dot_product / (magnitude_AB * magnitude_BC), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_theta))  # Convert to degrees

def predictSign(data):
    landmarks = np.array(data[0])[:, 1:]  # Extract x, y coordinates as NumPy array

    feature_dict = {}  # Dictionary to store features before creating DataFrame

    num_points = landmarks.shape[0]  # Number of landmarks

    # Compute distances (vectorized)
    i_indices, j_indices = np.triu_indices(num_points, k=1)  # Get upper triangle indices
    distances = np.sqrt(
        (landmarks[i_indices, 0] - landmarks[j_indices, 0]) ** 2 + 
        (landmarks[i_indices, 1] - landmarks[j_indices, 1]) ** 2
    )

    # Store distance features
    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
        feature_dict[f"dist_{i}_{j}"] = distances[idx]

    # Compute angles (vectorized)
    i_indices, j_indices, k_indices = np.triu_indices(num_points, k=2)  # Get triplet indices
    A = landmarks[i_indices]
    B = landmarks[j_indices]
    C = landmarks[k_indices]

    angles = calculate_angles_vectorized(A, B, C)

    # Store angle features
    for idx, (i, j, k) in enumerate(zip(i_indices, j_indices, k_indices)):
        feature_dict[f"angle_{i}_{j}_{k}"] = angles[idx]

    # Convert dictionary to DataFrame
    df2 = pd.DataFrame([feature_dict])

    # Predict using the model
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
    
    
    