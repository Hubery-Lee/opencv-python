# -*- coding: utf-8 -*-
# @Time    : 2020/7/28 9:55
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## 文档识别
## 参考chapter 4 仿生变换

import numpy as np
import matplotlib as plt
import  cv2

"""
思路：
1.检测区域
2.图像分割
3.仿生变换
4.结果展示
5.文字识别
"""

def order_points(pts):
    """
    对变换对象的几何顶点进行排序
    :param pts: 输入坐标点列表
    :return: 返回排序后的坐标点
    """
    # print(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    # print("s")
    # print(s)
    # print(np.argmin(s))
    # print(np.argmax(s))

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_points_transform(img,pts):
    """
    仿生变换
    :param img: 输入包含待变换对象的图像
    :param pts: 输入待变换对象在图像中的坐标点
    :return: 输出变换对象的图像
    """
    # 变换前的坐标
    org_pts = order_points(pts)
    # 变换后的坐标
    widthA = np.sqrt((org_pts[0][0]-org_pts[1][0])**2 + (org_pts[0][1]-org_pts[1][1])**2)
    widthB = np.sqrt((org_pts[2][0] - org_pts[2][0]) ** 2 + (org_pts[3][1] - org_pts[3][1]) ** 2)
    width = max(int(widthA),int(widthB))

    heightA = np.sqrt((org_pts[0][0]-org_pts[1][0])**2 + (org_pts[3][1]-org_pts[3][1])**2)
    heightB = np.sqrt((org_pts[1][0] - org_pts[2][0]) ** 2 + (org_pts[1][1] - org_pts[2][1]) ** 2)
    height = max(int(heightA),int(heightB))

    Points = np.array([[0, 0], [width-1,0],[width-1,height-1],[0,height-1]],dtype= "float32")

    # 仿生变换
    matrix2 = cv2.getPerspectiveTransform(org_pts,Points)
    imgWarp = cv2.warpPerspective(img,matrix2,(width,height))
    return imgWarp

def resize(img,height = None,width = None,inter = cv2.INTER_AREA):
    """
    对图像大小进行插值缩小或放大
    :param img: 待缩放图像
    :param height: 图像高度
    :param width: 图像宽度
    :param inter: openCV中的缩放插值方法
    :return: 缩放后的图像
    """
    dim = None
    (h,w) = img.shape[:2]
    if height is None and width is None:
        return img
    elif width is None:
        ratio = height/float(h)
        dim = (int(w*ratio),height)
    else:
        ratio = width/float(w)
        dim = (width,int(h*ratio))
    #  cv2.resize(img,(width,height),flag)
    resized = cv2.resize(img,dim,inter)
    return resized

# 读入图像
img = cv2.imread("page.jpg")
# 图像大小归一化
org = img.copy()
image = resize(img,height = 500)

# 图像预处理
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(blur,100,255)

# 轮廓检测
binary,contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#contours = contours[:5]
contours = sorted(contours,key = cv2.contourArea,reverse= True)[:5]

for cnt in contours:
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.02*perimeter
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
        screenCnt = approx
    break

# 画出检测区域
cv2.drawContours(img,[screenCnt],0,(0,0,255),2)
pts = screenCnt.reshape(4,2)

# 仿生变换
img = four_points_transform(img,pts)

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",img)
cv2.waitKey(0)

# 文字识别
## 需要安装tesseract-ocr
## 并配置好环境变量

from PIL import Image
import pytesseract
import os

text = pytesseract.image_to_string(img)
print(text)

output = 'img2string.txt'
with open(output,'w') as file_obj:
    file_obj.write(text)