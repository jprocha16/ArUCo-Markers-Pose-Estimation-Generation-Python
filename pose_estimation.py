'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, arucoid):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    tvec = np.zeros((1,1,3))
    # corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
    #     cameraMatrix=matrix_coefficients,
    #     distCoeff=distortion_coefficients)
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
    cv2.aruco.detectMarkers

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            if ids[i, 0] == arucoid:
                # Estimate pose of each marker and return the values rvec and tvec---(different
                # from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.16, matrix_coefficients,
                                                                               distortion_coefficients)
                print('rotation vector: ', rvec)
                print('translation vector: ', tvec)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)
                # Draw Axis
                # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
    return frame, tvec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument("-i", "--camera", type=bool, default=False, help="Set to True if using webcam")
    ap.add_argument("-v", "--video", help="Path to the video file")
    ap.add_argument("-id", "--ident", help="Desired ArUco marker id")
    args = vars(ap.parse_args())

    if args["camera"]:
        print('trying to open camera')
        video = cv2.VideoCapture(0)
        time.sleep(2.0)
    else:
        print('trying to open video file')
        if args["video"] is None:
            print("[Error] Video file location is not provided")
            sys.exit(1)

        video = cv2.VideoCapture(args["video"])
        if video is None or not video.isOpened():
            print('ERROR: video not found!')

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    aruco_id = int(args["ident"])

    time.sleep(0.1)
    cv2.namedWindow('Estimated Pose', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        h, w, _ = frame.shape
        # width = 1000
        # height = int(width * (h / w))
        #
        # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        z_list = []
        output, t_vec = pose_estimation(frame, aruco_dict_type, k, d, aruco_id)
        z_list.append(t_vec[0][0][2])
        cv2.imshow('Estimated Pose', output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()