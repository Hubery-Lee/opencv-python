# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 21:41
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com
import cv2

## 拼图

import numpy as np
img = cv2.imread("liudehua.jpg")
# join image 拼接图像
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

#转灰度图
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#二值化图
ret, imgBinary = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
# 高斯模糊
kernel = np.ones((5,5),np.uint8)
Blur = cv2.GaussianBlur(imgGray,(5,5),1)
# 边缘检测
imgCanny = cv2.Canny(imgGray,127,255)

# 膨胀 dialation
imgDilation = cv2.dilate(imgCanny,kernel)

# 腐蚀 errsion
imgErode = cv2.erode(imgDilation,kernel)

# imgArray = np.hstack([imgGray,imgCanny])
# imgArray2 = np.vstack([imgBinary,imgErode])
#
# cv2.imshow("hstack",imgArray)
# cv2.imshow("vstack",imgArray2)

imgArray = ([imgGray,imgCanny],[imgBinary,imgErode])
results = stackImages(0.5,imgArray);

cv2.imshow("results",results)
cv2.waitKeyEx(0)