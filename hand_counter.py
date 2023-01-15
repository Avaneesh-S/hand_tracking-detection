import cv2 as cv
import mediapipe as mp
import time

#using laptop camera:
cap=cv.VideoCapture(0)

#initialising mediapipe hand object
mpHands=mp.solutions.hands

#parameters: static_image_mode if its False then it will track and detect when confidence lvl drops,max_num_hands,min_detection_confidence,min_tracking confidence
#-->using default parameters :(False,2,0.5,0.5)
hands=mpHands.Hands()
#NOTE: hands object uses only 'RGB' images therefore need to convert BRG to RGB

#using mp inbuilt module for drawing on hand:
mp_draw=mp.solutions.drawing_utils


while True:
    success,img=cap.read()

    #converting to RGB:
    img_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #processing entire image to detect and track the palm:
    results=hands.process(img_rgb)

    flag=0
    hmap={4:0,3:0,8:0,6:0,12:0,10:0,16:0,14:0,20:0,18:0}
    #results.multi_hand_landmarks has the landmarks for every hand it detects in frame ,now we are drawing those landmarks and connections on image:
    if(results.multi_hand_landmarks):
        for hand in results.multi_hand_landmarks:
            for id,lm in enumerate(hand.landmark):
                #id is basically the landmark number and lm is the x,y coordinate of that landmark.
                #lm values are not in pixels , its as ratio, so finding pixel values:
                height,width,channels=img.shape
                x,y=int(lm.x * width),int(lm.y * height)
                #print(id)
                hmap[id]=y
                cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                #NOTE:there are total 21 landmarks in a palm (0-20), we are drawing each and their connections
                mp_draw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)

    count=0
    if(hmap[4]<=hmap[3] and hmap[4]!=0 and hmap[3]!=0):
        count+=1
    if(hmap[8]<=hmap[6] and hmap[8]!=0 and hmap[6]!=0):
        count+=1
    if(hmap[12]<=hmap[10] and hmap[12]!=0 and hmap[10]!=0):
        count+=1
    if(hmap[16]<=hmap[14] and hmap[16]!=0 and hmap[14]!=0):
        count+=1
    if(hmap[20]<=hmap[18] and hmap[20]!=0 and hmap[18]!=0):
        count+=1

    cv.putText(img, str(count),(100,100), cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 2)


    cv.imshow("image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()