# -*- coding: utf-8 -*-
# @Time    : 2020/7/28 10:10
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## 答题卡识别

'''
0. 答题卡仿生变换
1. 图像预处理
2. 形态学处理
3. 找出涂黑的选项
4. 与答案进行匹配
'''
import cv2
import numpy as np
import matplotlib as plt

# test01_png的答案
ANSWER = {0:1, 1:4, 2:0, 3:2, 4:1}

def cv_imshow(winname, src):
    """
    显示图片，按任意键关闭
    :param winname: 窗口名字
    :param src: 待显示图片源文件
    :return: 无返回值
    """
    cv2.imshow(winname, src)
    cv2.waitKey(0)


def cnts_sorted(cnts):
    """
    画边框，按X轴坐标位置排序
    :param cnts: 输入等高线组
    :return: 排序后的等高线组和boundingBoxes（外接矩形组）
    """
    boudaryBoxes = [cv2.boundingRect(cnt) for cnt in cnts]
    dat = zip(cnts, boudaryBoxes)
    (Cnts, Boxes) = zip(*sorted(dat, key=lambda b: b[1][1],reverse=False))
    return Cnts, Boxes

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

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
image = cv2.imread("test_01.png")
# image = resize(image,500)
img = image.copy()
# 预处理
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(img,100,255)
cv_imshow("Canny",edges)

# 找出答题卡区域
# 1. 检测轮廓
# 2. 提取坐标
# 3. 仿生变换变换

binary, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key = cv2.contourArea,reverse= True)[:5] #

for cnt in contours:
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.02*perimeter
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    if len(approx) == 4:
        screenCnt = approx
    break

cv2.drawContours(image,[screenCnt],0,(0,0,255),2)
cv_imshow("screen",image)

pts = screenCnt.reshape(4,2)
warped = four_points_transform(img,pts)
cv_imshow("warped",warped)

# 找出涂黑的选项位置
# 1. 二值化处理
# 2. 检测圆圈轮廓
# 3. 筛选涂黑选项
cv_imshow("warped_",warped)
ret, thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv_imshow("Ostu", thresh)

bin_c, cnts_c, hierarchy_c = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
th = thresh.copy()
# warped = cv2.cvtColor(warped,cv2.COLOR_GRAY2BGR)
cv2.drawContours(th, cnts_c, -1, (0, 0, 255), 2)
cv_imshow("Ostu_cnts",th)

# 选项轮廓
questionCnts = []
# 遍历
for c in cnts_c:
    # 计算比例和大小
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # 根据实际情况指定标准
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# 按照从上到下进行排序
# questionCnts, _ = cnts_sorted(questionCnts)
questionCnts,_ = sort_contours(questionCnts, method="top-to-bottom")

# print(questionCnts)
# warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
# for i,cnt in enumerate(questionCnts):
#     (x,y,w,h) = cv2.boundingRect(cnt)
#     cv2.drawContours(warped, cnt, 0, (0, 0, 255), 2)
#     cv2.putText(warped,str(i),(x-10,y-10),cv2.FONT_ITALIC,0.5,(0,0,255),2)
#     cv_imshow("order", warped)

correct = 0

warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
# 每排有5个选项
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # 排序
    cnts, _ = sort_contours(questionCnts[i:i + 5])
    bubbled = None

    # 遍历每一个结果
    for (j, c) in enumerate(cnts):
        # 使用mask来判断结果
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)  # -1表示填充
        # cv_imshow('mask', mask)
        # 通过计算非零点数量来算是否选择这个答案
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # cv_imshow("maskbit", mask)
        # print("total")
        # print(total)

        # 通过阈值判断
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    print(bubbled)
    # 对比正确答案
    color = (0, 0, 255)
    k = ANSWER[q]

    # 判断正确
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # 绘图
    cv2.drawContours(warped, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)



