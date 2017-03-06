from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file")
args = vars(ap.parse_args())

if not args.get("video", False):
	camera = cv2.VideoCapture(0)

else:
	camera = cv2.VideoCapture(args["video"])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#count = 0
count_up = 0
count_down = 0

centroid_init, centroid_current = [], []
init = 0

while True:
	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame, width=300)
	orig = frame.copy()

	(rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)

	for (x,y,w,h) in rects:
		cv2.rectangle(orig, (x,y), (x+w, y+h), (0,0,255), 2)

	rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	centroid_current[:] = []
	
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(frame, (xA,yA), (xB,yB), (0,255,0), 2)
		if init != 0:
			centroid_current.append(((xA + xB) * 0.5, (yA + yB) * 0.5))
		else:
			centroid_init.append(((xA + xB) * 0.5, (yA + yB) * 0.5))

	if init == 0:
		init = 1

	#print(centroid_init)

	xy1 = np.asarray(centroid_init)
	xy2 = np.asarray(centroid_current)

	if xy1.all() != xy2.all():
		print('0')
	print(xy1, xy2)
	#print(xy1, xy2)

	if len(xy2) > 0 and len(xy1) > 0:
		if len(xy2) > 1:
			dists = np.sqrt((xy1[:, 0, np.newaxis] - xy2[:, 0])**2 + (xy1[:, 1, np.newaxis] - xy2[:, 1])**2)
			mindist = np.min(dists, axis=1)
			minid = np.argmin(dists, axis=1)
			displacement = xy2[minid] - xy1
			displacement = displacement[:, 0]
		elif len(xy2) == 1 and len(xy1) > 1:
			dists = np.sqrt((xy1[0, np.newaxis] - xy2[:, 0])**2 + (xy1[1, np.newaxis] - xy2[:, 1])**2)
			mindist = np.min(dists, axis=1)
			minid = np.argmin(dists, axis=1)
			displacement = xy2 - xy1[minid]
			displacement = displacement[0]
		elif len(xy2) == 1 and len(xy1) == 1:
			dists = np.sqrt((xy1[0,0] - xy2[0,0])**2 + (xy1[0,1] - xy2[0,1])**2)
			displacement = xy2 - xy1
			displacement = displacement[0]
		i = 0
		for (x, y) in centroid_current:
			i = i+1
			if 148 <= x < 150:
				if displacement[0] > 0:
					count_up = count_up + 1
				else:
					count_down = count_down + 1
	else:
		pass

	#print(displacement)
	
    

	#cv2.imshow("Before NMS", orig)
	cv2.putText(frame, "Count_up: {}".format(count_up), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, "Count_down: {}".format(count_down), (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.imshow("After NMS", frame)

	#if init % 3 == 0:
	#centroid_init[:] = []
	centroid_init = centroid_current.copy()
	

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
