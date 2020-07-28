# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 17:49
# @Author  : Hubery-Lee  
# @Email   : hrbeulh@126.com

# 人脸检测
import cv2

# faceCascade_name = "haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier()
# faceCascade.load(faceCascade_name)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img = cv2.imread("liudehua.jpg")
cap = cv2.VideoCapture(0)

while True:
    sucess,img = cap.read()
    faces = faceCascade.detectMultiScale(img)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("face", img)
    if cv2.waitKeyEx(1)& 0xFF == ord('q'):
        break
