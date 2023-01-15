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

    cv.putText(img,"VOLUME",(80,60),cv.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255),2)
    cv.rectangle(img, (80, 100), (150, 400), (255, 0, 0))
    cv.putText(img, "100%", (15, 120), cv.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 2)
    cv.putText(img, "0%", (30, 395), cv.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 2)
    #converting to RGB:
    img_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #processing entire image to detect and track the palm:
    results=hands.process(img_rgb)

    #results.multi_hand_landmarks has the landmarks for every hand it detects in frame ,now we are drawing those landmarks and connections on image:
    if(results.multi_hand_landmarks):
        thumb_x, thumb_y, index_x, index_y = -1, -1, -1, -1
        for hand in results.multi_hand_landmarks:

            for id,lm in enumerate(hand.landmark):
                #id is basically the landmark number and lm is the x,y coordinate of that landmark.
                #lm values are not in pixels , its as ratio, so finding pixel values:
                height,width,channels=img.shape
                x,y=int(lm.x * width),int(lm.y * height)
                #print(id)

                if(id==4 ):
                    thumb_x=x
                    thumb_y=y
                    cv.circle(img,(x,y),20,(0,0,255),cv.FILLED)

                if (id == 8):
                    index_x = x
                    index_y = y
                    cv.circle(img, (x, y), 20, (0, 0, 255), cv.FILLED)

                #NOTE:there are total 21 landmarks in a palm (0-20), we are drawing each and their connections
                mp_draw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)

            cv.line(img, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255))
            mid_x=(thumb_x+index_x)//2
            mid_y = (thumb_y + index_y) // 2
            cv.circle(img,(mid_x,mid_y),20,(0,0,255),cv.FILLED)

            dist=int(((index_x-thumb_x)**2+(index_y-thumb_y)**2)**0.5)
            new_y=400-int(dist*1.2)
            if(new_y<100):
                new_y=100
            if(new_y>350):
                new_y=400
            cv.rectangle(img, (80,new_y), (150, 400), (255, 0, 0),thickness=-1)

#270-285 , 19-25
    cv.imshow("image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()