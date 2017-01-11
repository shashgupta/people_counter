from __future__ import print_function
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

cap = cv2.VideoCapture('VID2.mp4')
# params for ShiTomasi corner detection
#feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

 # Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(old_frame, winStride=(4,4), padding=(8,8), scale=1.05)
rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
centroid_current = []
for (xA, yA, xB, yB) in pick:
    centroid_current.append(((xA + xB) * 0.5, (yA + yB) * 0.5))

p0 = np.asarray(centroid_current, dtype = np.float32)[:, None]
print(p0.shape)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
     ret,frame = cap.read()
     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

     #print(err)
   # Select good points
     #print(st.shape)
     #print(p1.shape)
     good_new = p1[st==1]
     good_old = p0[st==1]
 
  # draw the tracks
     for i,(new,old) in enumerate(zip(good_new,good_old)):
         a,b = new.ravel()
         c,d = old.ravel()
         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
     img = cv2.add(frame,mask)
     cv2.imshow('frame',img)
     k = cv2.waitKey(30) & 0xff
     if k == 27:
         break
   
       # Now update the previous frame and previous points
     old_gray = frame_gray.copy()
     p0 = good_new.reshape(-1,1,2)
    
cv2.destroyAllWindows()
cap.release()