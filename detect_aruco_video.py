'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
ap.add_argument("-o", "--save", type=bool, default=True, help="save video or not")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
	video = cv2.VideoCapture(0)
	time.sleep(2.0)
	
else:
	if args["video"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

	video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

if args["save"]:
	print("save output")

	# showing values of the properties
	print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
	print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	print("CAP_PROP_FPS : '{}'".format(video.get(cv2.CAP_PROP_FPS)))

	w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = w
	height = int(width * (h / w))
	out_video = cv2.VideoWriter('Images/output_video.avi', 0,
								30,
								(w, h))

while True:
	ret, frame = video.read()
	
	if ret is False:
		break

	h, w, _ = frame.shape

	# width = 1000
	# height = int(width*(h/w))
	#
	# frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	start_time = time.time()
	corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

	print('processing time: ', time.time()-start_time)
	detected_markers = aruco_display(corners, ids, rejected, frame)

	if args["save"]:
		out_video.write(detected_markers)


	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
video.release()

if args["save"]:
	out_video.release()
