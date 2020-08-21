# -*- coding: utf-8 -*-
# @Time    : 2020/7/28 10:11
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## 目标追踪
import cv2
import matplotlib as plt
import numpy as np

'''
1. 选择追踪算法
2. 读入视频数据
3. 选择追踪目标
'''
# opencv已经实现了的追踪算法
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# 实例化OpenCV's multi-object tracker
trackers = cv2.MultiTracker_create()

# 读取视频
wide = 680
high = 450
cap = cv2.VideoCapture("soccer_01.mp4")
while True:
    sucess, img = cap.read()
    if img is None:
        break

    img = cv2.resize(img, (wide, high))
    # cv2.imshow("Video",img)
    # if cv2.waitKeyEx(1)& 0xFF == ord('q'):
    #     break

    # 追踪结果
    (success, boxes) = trackers.update(img)

    # 绘制区域
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示
    cv2.imshow("Frame", img)
    key = cv2.waitKey(80) & 0xFF

    if key == ord("s"):
        roi_box = cv2.selectROI("Frame",img)
        tracker = OPENCV_OBJECT_TRACKERS['boosting']()
        trackers.add(tracker,img,roi_box)
    elif key== 27:
        break
