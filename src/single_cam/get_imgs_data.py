import os
import glob
import cv2 as cv
import numpy as np


def get_images(video_folder: str, fps: int):
    imgs_folder = os.getcwd() + os.sep + "Single_Video_Images_data"
    os.makedirs(imgs_folder, exist_ok=True)
    video_path = glob.glob(os.getcwd() + os.sep + video_folder + os.sep + "*.mp4")[0]
    cap = cv.VideoCapture(video_path)
    frame_number = 0
    tmp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        elif tmp >= 99999:
            break

        if tmp % fps != 0:
            tmp += 1
        else:
            img_name = f"frame_{frame_number}.jpg"
            cv.imwrite(imgs_folder + os.sep + img_name, frame)
            tmp += 1
            frame_number += 1
    print("Done!")


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

    print(mtx)
    print()
    print(dist)


if __name__ == "__main__":
    chessboard_size = (10, 7)
    chessboard_len = (25, 1)
    imgs_folder = "Single_Video_Images_data"
    # chessboard_video_detection("Video_Data", chessboard_size)
    get_images("Video_Data", 20)
    chessboard_images_detection(imgs_folder, chessboard_size, chessboard_len)



