'''
Sample Usage:
python calibration.py -i v -d ./Images/calibration_video.mp4  -s 0.022 -v true
'''

import numpy as np
import cv2
import os
import argparse
import shutil

def calibrate(dirpath, square_size, width, height, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dirpath)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

        if visualize:
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow('img',img)
            cv2.waitKey(10)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="type of input: video or images (v or i)")
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)",  default=9)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float, default=1, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    dirpath = args['dir']
    # 2.4 cm == 0.024 m
    # square_size = 0.024
    square_size = args['square_size']

    width = args['width']
    height = args['height']

    input_type = args['input']
    visualize = bool(args['visualize'])

    if input_type == 'v':
        video = cv2.VideoCapture(args["dir"])

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)

        #check existence of a folder called temp_image_calibration
        isExist = os.path.exists(os.path.split(args["dir"])[0] + '/temp_image_calibration')
        if isExist:
            print('folder: temp_image_calibration already exists. Delete folder to proceed')
            exit()
        else:
            os.chdir(os.path.split(args["dir"])[0])
            os.mkdir('temp_image_calibration')

            ctr = 1
            while True:
                ret, frame = video.read()

                if ret is False:
                    print('error reading frame')
                    break
                h, w, _ = frame.shape

                if args['visualize']:
                    cv2.imshow('output', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                filename = 'temp_image_calibration/' + str(str(ctr).zfill(5)) + '.png'
                if not np.remainder(ctr, 5):
                    cv2.imwrite(filename, frame)
                ctr = ctr + 1

            os.chdir('../')

        dirpath = os.path.split(args["dir"])[0] + '/temp_image_calibration'
    else:
        pass

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False

    ret, mtx, dist, rvecs, tvecs = calibrate(dirpath, square_size, visualize=visualize, width=width, height=height)

    print(mtx)
    print(dist)

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)

    os.chdir(os.path.split(args["dir"])[0])

    cv2.destroyAllWindows()
    shutil.rmtree('temp_image_calibration')
