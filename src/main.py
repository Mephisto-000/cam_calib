import os
from single_cam.get_imgs_data import get_images
from single_cam.get_mtx_dist import chessboard_images_detection
from stream_board_check import chessboard_stream_detection as ch_b_stream


def main():
    chessboard_size = (10, 7)
    chessboard_len = (25, 1)
    video_folder_name = "Video_Data"
    imgs_folder_name = "Single_Video_Images_data"
    system_choose = input("Please input your OS : Windows or Linux? [w/l] ")
    while True:
        if system_choose == "w":
            os.system("cls")
        elif system_choose == "l":
            os.system("clear")
        else:
            print("Error OS input !")

        print("To test real-time chessboard detection, please enter 'c1'.")
        print("To test extracting multiple images from a recorded video, please enter 'c2'.")
        print("To test detecting the chessboard in the captured images and calculate the camera matrix and distortion "
              "coefficients, please enter 'c3'.")

        command = input("Please input your test case: ")
        if command == "q":
            break

        if command == "c1":
            ch_b_stream(chessboard_size)
        elif command == "c2":
            get_images(video_folder_name, 20)  # per 20 fps
        elif command == "c3":
            chessboard_images_detection(imgs_folder_name, chessboard_size, chessboard_len)


if __name__ == "__main__":
    main()

