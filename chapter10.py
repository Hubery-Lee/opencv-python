# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 20:54
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## virtual paint 虚拟油画

# 1. 检测颜色
# 2. 检测等高线区域,计算外接矩形，获取画笔落笔点坐标
# 3. 画图

import cv2
import numpy as np

def stackImages(scale,imgArray):
    """
    1.计算图像元组的的行列数 rows cols
    2.计算图像的宽和高 width height
    :param scale: 缩放比例
    :param imgArray: tuple 图像元组，包括图像列表
    :return: np.hstack np.vstack
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    '''
    isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
    isinstance() 与 type() 区别：
    type() 不会认为子类是一种父类类型，不考虑继承关系。
    isinstance() 会认为子类是一种父类类型，考虑继承关系。
    如果要判断两个类型是否相同推荐使用 isinstance()。
    '''
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

def empty(a):
    pass

def findColor(imgLocal):
    """
    检出颜色
    :param img:
    :return:
    """
    imgLocalHSV = cv2.cvtColor(imgLocal, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lowerb = np.array([h_min, s_min, v_min])
    upperb = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgLocalHSV, lowerb, upperb)
    # https://www.geeksforgeeks.org/arithmetic-operations-on-images-using-opencv-set-2-bitwise-operations-on-binary-images/
    #imgResult = cv2.bitwise_and(imgLocal, imgLocal, mask=mask)

    return mask


def getContours(img):
    """
    检出边缘
    :param img: mask
    :return: Point position
    """
    binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 400:
            #cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 2)
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.01 * perimeter  # 精度参数
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgLocal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgLocal, "Pen", (x + (w // 2) - 20, y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0., 255, 0), 1)
    return x+w//2, y


def drawPoints(Points):
    """
    :param img: input imgLocal
    :return: modified image
    """
    for point in Points:
        cv2.circle(imgLocal,(point[0], point[1]),10,(0,0,255),cv2.FILLED)

cap = cv2.VideoCapture(0)

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 108, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 31, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 200, 255, empty)

myColor =[[0.,179,108,255,31,200]]
Points = []
while True:
    sucess, img = cap.read()
    imgLocal = img.copy()
    imgColArea = findColor(imgLocal)
    # imgContour = np.ones_like(imgColArea)
    x, y = getContours(imgColArea)

    if x!=0 and y!=0:
        Points.append([x, y])
    # print(Points)
    drawPoints(Points)

    # cv2.imshow("img",img)
    # cv2.imshow("imgCol",imgColArea)
    # cv2.imshow("imgLocal",imgLocal)
    imgResult = stackImages(0.8,[img,imgColArea,imgLocal])
    cv2.imshow("imgResult",imgResult)

    if cv2.waitKey(1) == ord('q'):
        break
