import cv2
import numpy as np
img = cv2.imread("liudehua.jpg")
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

# 截取 crop
print(imgGray.shape)
imgCrop = imgGray[200:500,100:300]

cv2.imshow("origin",img)
cv2.imshow("gary",imgGray)
cv2.imshow("binary",imgBinary)
cv2.imshow("blur",Blur)
cv2.imshow("Canny",imgCanny)
cv2.imshow("Dilation",imgDilation)
cv2.imshow(
    "Errosion",imgErode
)
cv2.imshow("Crop",imgCrop)
cv2.waitKeyEx(0)

