# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 14:56
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## 检测形状

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

def getContours(img):
    """
    1. findContours()
    2. 利用面积去除背景噪声，提取感兴趣区域
    3. 获取角点数，利用其判断该Contour的形状
    4. 画出该形状的外接矩形，并添加该形状的识别标签
    :param img: edge detection
    :return: void
    """
    binary,contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 400:
            cv2.drawContours(imgContour,cnt,-1,(0,0,255),2)

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.01*perimeter #精度参数
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        print(approx) # 描述轮廓各点的坐标集
        print(len(approx)) #轮廓顶点数
        ObjPt = len(approx)
        x,y,w,h = cv2.boundingRect(approx)
        if ObjPt== 3: objType = "Triangle"
        elif ObjPt == 4:
            if w/float(h)>0.95 and w/float(h)<1.05: objType = "Square"
            else: objType = "Rectangle"
        elif ObjPt > 4:  objType = "Circle"
        else: None

        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(imgContour,objType,(x+(w//2)-20,y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0.,255,0),1)

img =cv2.imread("shapes.png")

# BGR2GRAY
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# BLUR
imgGaussian = cv2.GaussianBlur(imgGray,(7,7),1)
# EDGE DETECTION
imgCanny = cv2.Canny(imgGaussian,50,255)
# DRAW CONTOURS AND ADD PAINTXT
imgContour = img.copy()
# imgContour = np.zeros_like(img)
getContours(imgCanny)

# cv2.imshow("img", img)
# cv2.imshow("imgBlur", imgGaussian)
# cv2.imshow("imgCanny", imgCanny)
# cv2.imshow("imgContour", imgContour)

# 合并图像
def imgFormat(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

imgVis = np.zeros((2*img.shape[0],2*img.shape[1],3),np.uint8)
imgVis[0:img.shape[0],0:img.shape[1]] = img
imgVis[img.shape[0]:2*img.shape[0],0:img.shape[1]] = imgFormat(imgGaussian)
imgVis[0:img.shape[0],img.shape[1]:2*img.shape[1]] = imgFormat(imgCanny)
imgVis[img.shape[0]:2*img.shape[0],img.shape[1]:2*img.shape[1]] = imgContour
imgVis = cv2.resize(imgVis,(800,800))

# imgBlank = np.zeros_like(img)
# #imgVis = stackImages(0.6,([img,imgGray,imgGaussian],[imgCanny,imgContour,imgBlank]))
# imgVis = stackImages(0.6,[[img,imgGray,imgGaussian],[imgCanny,imgContour,imgBlank]])
cv2.imshow("results",imgVis)

cv2.waitKeyEx(0)