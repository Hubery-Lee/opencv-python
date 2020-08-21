# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 21:40
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

# 背景建模
"""
帧差法
由于场景中的目标在运动，目标的影像在不同图像帧中的位置不同。该类算法对时间上连续的两帧图像进行差分运算，不同帧对应的像素点相减，判断灰度差的绝对值，当绝对值超过一定阈值时，即可判断为运动目标，从而实现目标的检测功能。
"""
import numpy as np
import cv2

# 经典的测试视频
cap = cv2.VideoCapture('test.avi')
# 形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

while (True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        if perimeter > 188:
            # 找到一个直矩形（不会旋转）
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
