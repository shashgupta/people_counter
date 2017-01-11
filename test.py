import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

'''while(cap.isOpened()):  # check !
    # capture frame-by-frame
    ret, frame = cap.read()

    if ret: # check ! (some webcam's need a "warmup")
        # our operation on frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break'''
# When everything is done release the capture
