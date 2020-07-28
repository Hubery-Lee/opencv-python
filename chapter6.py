# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 8:00
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

import cv2
import numpy as np
import matplotlib as plt
# 全景图像拼接  stitch

class Stitch:
    def cv_show(self, name, img):
        """
        show image
        :param name:
        :param img:
        :return:
        """
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Stitch(self,img1,img2):
        """
        stitch images
        1.利用SIFT找到关键特征点
        2.特征点匹配
        3.根据特征点配匹位置，找到对应变换矩阵
        4.仿生变换
        :param img1: src1
        :param img2: src2
        :return:
        """
        self.img1=img1
        self.img2=img2

        ## detect features and describe features

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 初始化SIFT描述符
        sift = cv2.xfeatures2d.SIFT_create()
        # 基于SIFT找到关键点和描述符
        kps1, features1 = sift.detectAndCompute(gray1, None)
        kps2, features2 = sift.detectAndCompute(gray2, None)
        # 将结果转换成NumPy数组
        kpsA = np.float32([kp.pt for kp in kps1])
        kpsB = np.float32([kp.pt for kp in kps2])

        ## features match

        # 默认参数初始化BF匹配器
        bf = cv2.BFMatcher()
        rawMatches = bf.knnMatch(features1, features2, k=2)
        # 应用比例测试
        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        good = []
        for mm, n in rawMatches:
            if mm.distance < 0.75 * n.distance:
                good.append([mm])

        ## draw featrues
        # cv.drawMatchesKnn将列表作为匹配项。
        img3 = cv2.drawMatchesKnn(img1,kps1,img2,kps2,good,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("match points",img3),cv2.waitKey(0)
        # return img3

        # 当筛选后的匹配对大于4时，计算视角变换矩阵 ?????????????
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4.0)

        # 将图片A进行视角变换，result是变换后图片
        result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        self.cv_show('result', result)
        # 将图片B传入result图片最左端
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
        self.cv_show('result', result)


imgA = cv2.imread("right_01.png")
imgB = cv2.imread("left_01.png")

aStitch =Stitch()
imgStitch = aStitch.Stitch(imgA,imgB)


#################
##  视角匹配矩阵还存在问题！！！！！