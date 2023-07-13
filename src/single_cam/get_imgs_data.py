import os
import glob
import cv2 as cv


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


# if __name__ == "__main__":
#     chessboard_size = (10, 7)
#     chessboard_len = (25, 1)
#     imgs_folder = "Single_Video_Images_data"
#     get_images("Video_Data", 20)
