import cv2
import numpy as np
img = cv2.imread('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\0421.png') #原圖
img_inv = cv2.imread('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\taichung_ipm_0421(3)(4).jpg')
testttt = cv2.getPerspectiveTransform([[1124,543], [1260,543], [1260,575], [1124,575]], [[1124,543], [1260,543], [1293,578], [1140,575]]) 
img = cv2.warpPerspective(img, img_inv, img.shape[:2][::-1])


