import cv2 as cv


def chessboard_stream_detection(cb_point_size: tuple, cam_flag=0):
    cap = cv.VideoCapture(cam_flag)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        frame_copy = frame.copy()
        gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
        _, pts = cv.findChessboardCorners(gray, cb_point_size)
        det_frame = cv.drawChessboardCorners(frame, cb_point_size,
                                             pts, True)
        cv.imshow('chessboard detection', det_frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
