# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
def foto():
    img1 = cv2.imread(r'C:\Users\Admin\Desktop\ashtray.jpg')
    cv2.namedWindow('Displaywindow', cv2.WINDOW_NORMAL)
    cv2.imshow('Displaywindow', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    #cap = cv2.VideoCapture(r'C:\Users\Admin\Desktop\video_2022-09-17_12-04-24.mp4')
    while True:
        ret, frame = cap.read()
        if not (ret):
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def readIPWriteTOFile():
    video = cv2.VideoCapture(0)
    ok, img = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("outputt.mov", fourcc, 25, (w, h))
    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def WebCam():
    image = cv2.VideoCapture(1)
    while True:
        rez, img = image.read()
        cv2.imshow('Camera', img)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:  # if cv2.waitKey(60) & OxFF == ord('q')
            break

    cv2.destroyAllWindows()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #foto()
    #ideo()
    #readIPWriteTOFile()
    WebCam()