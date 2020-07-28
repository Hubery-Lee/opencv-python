import cv2
import  numpy as np

# 作图
Canvas = np.zeros((500,500,3),np.uint8)
cv2.rectangle(Canvas,(10,10),(300,400),(0,0,255),2)
cv2.circle(Canvas,(400,450),10,(0,255,0),2)
cv2.line(Canvas,(0,0),(500,500),(0,204,102),3)
cv2.putText(Canvas,"OpenCV Hubery Lee",(200,300),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,0))

cv2.imshow("Canvas",Canvas)
cv2.waitKeyEx(0)