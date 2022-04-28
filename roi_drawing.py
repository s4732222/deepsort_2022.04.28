'''
#潭子 10m
line1 = [(554,770), (1095,770)]
line2 = [(554,863), (1095,863)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''
'''
#潭子 10m
line1 = [(445,801), (1110,801)]
line2 = [(430,883), (1117,883)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''
'''
#台灣大道朝富路
line1 = [(619,320), (883,320)]
line2 = [(619,378), (883,378)]
line3 = [(0,0), (0,0)]
line4 = [(0,0), (0,0)]
'''
'''
#測試用
line1 = [(429,583), (900,583)]
line2 = [(429,610), (900,610)]
line3 = [(959,575), (1490,576)]
line4 = [(958,545), (1486,545)]
'''
'''
#中清路曉明女中
line1 = [(430,587), (894,587)]
line2 = [(430,694), (894,694)]
line3 = [(995,515), (1458,515)]
line4 = [(995,575), (1460,575)]
'''
#台灣大道惠中路
line1 = [(288,619), (853,619)]
line2 = [(290,675), (865,678)]
line3 = [(981,641), (1878,642)]
line4 = [(981,713), (1878,711)]
# coding: utf-8
import cv2
import numpy as np

img = cv2.imread('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\taichung_ipm_0424(3)(4).jpg')





cv2.line(img, line1[0], line1[1], (0, 0, 255), 1)
cv2.line(img, line2[0], line2[1], (0, 0, 255), 1)
cv2.line(img, line3[0], line3[1], (0, 0, 255), 1)
cv2.line(img, line4[0], line4[1], (0, 0, 255), 1)
cv2.imshow('My Image', img)
cv2.imwrite('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\taichung_roi_ipm_0424(3)(4).jpg', img)        
cv2.waitKey(0)
cv2.destroyAllWindows()


