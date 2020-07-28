# import cv2
#
# def mouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 255, 255), thickness = -1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (255, 255, 255), thickness = 1)
#         cv2.imshow("image", img)
#
# img = cv2.imread("liudehua.jpg")
# # cv2.namedWindow("image")
# cv2.imshow("image", img)
# # cv2.resizeWindow("image", 800, 600)
# cv2.setMouseCallback("image", mouse)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# 参数解释
# image：源图像
# threshold1：阈值1
# threshold2：阈值2
# apertureSize：可选参数，Sobel算子的大小
# 其中，较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
# 函数返回的是二值图，包含检测出的边缘

import numpy as np
import cv2 as cv
cv.namedWindow("images")
def nothing():
    pass
cv.createTrackbar("s1","images",0,255,nothing)
cv.createTrackbar("s2","images",0,255,nothing)
img = cv.imread("left_01.png",0)
while(1):
    img = cv.imread("lambo.PNG")
    s1 = cv.getTrackbarPos("s1","images")
    s2 = cv.getTrackbarPos("s2","images")
    out_img = cv.Canny(img,s1,s2)
    cv.imshow("img",out_img)
    k = cv.waitKey(1)
    if k==ord("q"):
        break
cv.destroyAllWindows()

## 图像pingjie

# import cv2
# import numpy as np
#
# img = cv2.imread("lambo.PNG")
# imgBlur = cv2.GaussianBlur(img,(7,7),1)
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print(imgGray.shape)
# imgGray = cv2.cvtColor(imgGray,cv2.COLOR_GRAY2BGR)
# imgStick = np.hstack([img,imgBlur,imgGray])
#
# cv2.imwrite("img_test.png",imgStick)
#
# #cv2.imshow("imgBlur",imgBlur)
# # cv2.waitKeyEx(0)


# ## 全景图拼接
# import cv2
# import numpy as np
#
# imgA = cv2.imread("left_01.png")
# imgB = cv2.imread("right_01.png")
#
# GrayA = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
# GrayB = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kpsA, featuresA = sift.detectAndCompute(GrayA,None)
# kpsB, featuresB = sift.detectAndCompute(GrayB,None)
#
# kps1 = np.float32([kp.pt for kp in kpsB])
# kps2 = np.float32([kp.pt for kp in kpsA])
#
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(featuresA,featuresB,k = 2)
#
# mmatch = []
# for m in matches:
#     if len(m) == 2 and m[0].distance<0.75*m[1].distance:
#         mmatch.append((m[0].queryIdx,m[0].trainIdx))
#
# good = []
# for m,n in matches:
#     if m.distance<0.75*n.distance:
#         good.append([m])

# print(good.shape)
# img = cv2.drawMatchesKnn(GrayA,kpsA,GrayB,kpsB,good,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("img",img)
# cv2.waitKeyEx(0)

# img3 = cv2.drawMatchesKnn(GrayA, kpsA, GrayB, kpsB, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("match points", img3), cv2.waitKey(0)
#
#
# MIN_MATCH_COUNT =4
# if len(mmatch)>MIN_MATCH_COUNT:
#     ptsA = np.float32([kps1[i] for (_, i) in mmatch])
#     ptsB = np.float32([kps2[i] for (i, _) in mmatch])
#
#     # 计算视角变换矩阵
#     (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
#
#     # 将图片A进行视角变换，result是变换后图片
# result = cv2.warpPerspective(imgB, H, (imgA.shape[1] + imgB.shape[1], imgB.shape[0]))
# cv2.imshow('result', result)
# # 将图片B传入result图片最左端
# result[0:imgA.shape[0], 0:imgA.shape[1]] = imgA
# cv2.imshow('result', result)
#
# cv2.waitKeyEx(0)