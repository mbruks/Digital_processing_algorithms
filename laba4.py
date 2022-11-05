
import cv2
import numpy as np


def webcam():
    video = cv2.VideoCapture(0)
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer1 = cv2.VideoWriter('input.mov', fourcc, 25, (w, h))
    video_writer2 = cv2.VideoWriter('output.mov', fourcc, 25, (w, h))

    cur_frame = None
    ok, img = video.read()
    old_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while (True):

        ok, img = video.read()
        cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img', cur_frame)
        video_writer1.write(img)

        # if old_frame == None:
        #     old_frame = cur_frame

        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        frame_diff = cv2.absdiff(cur_frame, old_frame)
        ret, thresh = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # image = cv2.drawContours(cur_frame,contours,-1,(0,0,255),3)
        # contour = cv2.contourArea(contours)
        # final = cur_frame.copy()
        # contours = sorted(contours, key = cv2.contourArea, reverse = True)# Draw the contour
        for i in contours:
            if cv2.contourArea(i) > 1000:
                video_writer2.write(img)
                break

        #         final = cv2.drawContours(final, i, contourIdx = -1,
        #                          color = (255, 0, 0), thickness = 2)
        # cv2.imshow('Display window', final)
        # cv2.waitKey(0)
        old_frame = cur_frame
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video.release()


if __name__ == '__main__':
    webcam()