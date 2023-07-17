import os
import glob
import cv2 as cv
import numpy as np


def chessboard_images_detection(imgs_folder: str, cb_point_size: tuple, cb_point_len: tuple):
    gridw, gridh = cb_point_size
    glupper, gllower = cb_point_len
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    grid_len = float(glupper)/float(gllower)
    objp = np.zeros((gridw*gridh, 3), np.float32)
    objp[:, :2] = np.mgrid[0:gridw, 0:gridh].T.reshape(-1, 2)*grid_len
    objpoints = []
    imgpoints = []

    imgs_detection_folder = os.getcwd() + os.sep + "Single_Video_Detected_data"
    os.makedirs(imgs_detection_folder, exist_ok=True)

    imgs_path = glob.glob(os.getcwd() + os.sep + imgs_folder + os.sep + "*.jpg")
    for img in imgs_path:
        frame = cv.imread(img)
        frame_copy = frame.copy()
        gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (gridw, gridh), None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_name = os.path.basename(img).split(".")[0]
            frame_name = f"{img_name}_Det.jpg"
            frame_out = cv.drawChessboardCorners(frame_copy,
                                                 (gridw, gridh), corners2[:15, :, :],
                                                 ret)
            cv.imwrite(imgs_detection_folder + os.sep + frame_name, frame_out)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,
                                                      (1080, 1920),
                                                      None, None)

    np.savetxt(os.getcwd() + os.sep + "mtx.txt", mtx)
    np.savetxt(os.getcwd() + os.sep + "dist.txt", dist)


# if __name__ == "__main__":
#     chessboard_size = (10, 7)
#     chessboard_len = (25, 1)
#     imgs_folder = "Single_Video_Images_data"
#     chessboard_images_detection(imgs_folder, chessboard_size, chessboard_len)
