import  cv2
import numpy as np

# 图像的旋转变换
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness = 1)
        cv2.imshow("image",img)
        cv2.imwrite("pos.jpg", img)

# 图片的变换 warp affine
img = cv2.imread("cards.jpg")
print(img.shape)
h, w ,c= img.shape
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
matrix1 = cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),90,0.8)
imgRes = cv2.warpAffine(img,matrix1,(w,h))


#感生 warp perspective
post1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
post2 = np.float32([[0,0],[250,0],[0,350],[250,350]])
matrix2 = cv2.getPerspectiveTransform(post1,post2)
imgWarp = cv2.warpPerspective(img,matrix2,(250,350))

# cv2.imshow("image",img)
# cv2.setMouseCallback("image", mouse)

cv2.imshow("image",img)
cv2.imshow("warpAffine",imgRes)
cv2.imshow("warpPerspective",imgWarp)
cv2.waitKeyEx(0)

cv2.waitKey(0)
cv2.destroyAllWindows()