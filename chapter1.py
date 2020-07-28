# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 0:09
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

import cv2

## 读取图片
# img = cv2.imread("liudehua.jpg")
#
# cv2.imshow("dehua",img)
# cv2.waitKeyEx(0)

## 读取视频
# wide =680
# high =450
# cap = cv2.VideoCapture("test.mp4")
# while True:
#     sucess, img = cap.read()
#     img = cv2.resize(img,(wide,high))
#     cv2.imshow("Video",img)
#
#     if cv2.waitKeyEx(1)& 0xFF == ord('q'):
#         break

## web camera

wide =680
high =450
cap = cv2.VideoCapture(0)
cap.set(3,wide)
cap.set(4,high)
while True:
    sucess, img = cap.read()
    #img = cv2.resize(img,(wide,high))
    cv2.imshow("Video",img)

    # if cv2.waitKeyEx(1)& 0xFF == ord('q'):
    #     break

    if cv2.waitKeyEx(1)== ord('q'):
        break