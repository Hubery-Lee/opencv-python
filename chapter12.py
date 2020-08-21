# -*- coding: utf-8 -*-
# @Time    : 2020/7/28 10:05
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

## OCR 车牌识别 信用卡号识别

'''
0. 模板准备
1. 形态学处理
2. 数字区域分割
3. 匹配模板，数字识别
'''

import numpy as np
import cv2
import matplotlib as plt


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    图像大小缩放归一化操作
    :param image: 待缩放图像
    :param width: 缩放宽度
    :param height: 缩放高度
    :param inter: 插值方式
    :return: 缩放后的图像
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


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
    (Cnts, Boxes) = zip(*sorted(dat, key=lambda b: b[1][0]))
    return Cnts, Boxes


# 读取模板
tmp = cv2.imread("ocr_a_reference.png")
ref = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
ret, bin_ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)

# 模板中数字的识别

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
# 返回的list中每个元素都是图像中的一个轮廓
ref_bin, ref_cnts, ref_hierarchy = cv2.findContours(bin_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(tmp, ref_cnts, -1, (0, 0, 255), 2)

# 画边框，按X轴坐标位置排序
# boudaryBox = [cv2.boundingRect(cnt) for cnt in ref_cnts]
# ref_dat = zip(ref_cnts,boudaryBox)
# (refCnts, BoundingBox) = zip(*sorted(ref_dat,key = lambda b:b[1][0]))
refCnts, Boxes = cnts_sorted(ref_cnts)

digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = bin_ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 每一个数字对应每一个模板
    digits[i] = roi

# 显示模板
# for i,dgt in enumerate(digits):
#     cv_imshow("image",digits[i])


#
# 读入待识别信用卡
#
imgR = cv2.imread("credit_card_01.png")
imgR = resize(imgR, width=300)
imgRes = imgR.copy()
img = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# 形态学处理
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
squaKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, rectKernel)
cv_imshow("tophat", tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # ksize=-1相当于用3*3的

gradX = np.absolute(gradX)  # 求绝对值
(minVal, maxVal) = (np.min(gradX), np.max(gradX))  # 找最小值最大值
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 归一化 或者说是直方图均衡化
gradX = gradX.astype("uint8")  # 数据类型转换

print(np.array(gradX).shape)
# cv_imshow("gradX",gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# cv_imshow("gradX",gradX)
# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
ret, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv_imshow("thresh",thresh)
# 再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squaKernel)
# cv_imshow("thresh",thresh)
# 计算轮廓
Binary, Cnts, Hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgR, Cnts, -1, (0, 0, 255), 3)
# cv_imshow("Contours",imgR)
locs = []
# 筛选轮廓
# for i,c in enumerate(Cnts):
for cnt in Cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = float(w) / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ratio > 2.5 and ratio < 4:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda a: a[0])
output = []
# 遍历每个轮廓中的数字
for gX, gY, gW, gH in locs:
    # initialize the list of group digits
    groupOutput = []
    # 根据坐标提取每一个组
    group = img[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_imshow("group", group)
    # 预处理
    # 自适应阈值处理
    ret, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv_imshow("group", group)
    # 计算每一组的轮廓
    bin, grp_cnts, g_hierarchy = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grpCnts, grpBox = cnts_sorted(grp_cnts)  # 按坐标位置排序
    # 计算每一组中的每一个数值
    for cnt in grpCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # cv_imshow('roi', roi)

        # 计算匹配得分
        scores = []

        # 模板匹配，在模板中计算每一个得分
        for (digit, digitROI) in digits.items():  # 这个地方为什么这么用得注意，这里digit是指点，（key,value）
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        # print(scores)
        groupOutput.append(str(np.argmax(scores)))
        # print(groupOutput)

    # 画出来
    cv2.rectangle(imgRes, (gX - 5, gY - 5),
                  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(imgRes, "".join(groupOutput), (gX, gY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card #: {}".format("".join(output)))
cv_imshow("result", imgRes)
