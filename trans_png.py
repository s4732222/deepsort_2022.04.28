import cv2
import numpy as np

'''
# 读取名称为 p19.jpg的图片
img = cv2.imread("C:/Users/sheng/Documents/My Image.jpg",1)
img_org = cv2.imread("C:/Users/sheng/Documents/My Image.jpg",1)

# 得到图片的高和宽
img_height,img_width = img.shape[:2]

# 定义对应的点
points1 = np.float32([[38,140], [96,145], [160,92], [187,94]])
points2 = np.float32([[0,0], [330,0], [0,330], [330,330]])

# 计算得到转换矩阵
M = cv2.getPerspectiveTransform(points1, points2)

# 实现透视变换转换
processed = cv2.warpPerspective(img,M,(330, 330))

# 显示原图和处理后的图像
cv2.imshow("org",img_org)
cv2.imshow("processed",processed)
cv2.imwrite("My Image_out.jpg",processed)
cv2.waitKey(0)
'''

img = cv2.imread('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\0424.png') #原圖
#img2 = cv2.imread('C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\taichung_ipm_0421.png') #轉置後的圖 用來反轉置

#pts = np.array([[1124,543], [1260,543], [1293,578], [1140,575]], dtype=np.float32)#中清路line34
#pts = np.array([[429,584], [896,559], [885,593], [389,612]], dtype=np.float32)#中清路line12
#pts = np.array([[430,587], [894,565], [879,665], [255,694]], dtype=np.float32)#中清路line12(new)
#pts = np.array([[1108,516], [1235,519], [1290,576], [1139,575]], dtype=np.float32)#中清路line34(new)
#pts = np.array([[562,711], [687,706], [610,818], [444,825]], dtype=np.float32)#台灣大道line12
#pts = np.array([[657,616], [750,615], [711,672], [596,677]], dtype=np.float32)#台灣大道line12(new)
pts = np.array([[1074,639], [1174,642], [1233,715], [1108,714]], dtype=np.float32)#台灣大道line34
for pt in pts:
    cv2.circle(img, tuple(pt.astype(np.int)), 1, (0,0,255), -1)

# compute IPM matrix and apply it

#ipm_pts = np.array([[1124,543], [1260,543], [1260,575], [1124,575]], dtype=np.float32)#中清路line34
#ipm_pts = np.array([[429,584], [896,584], [896,612], [429,610]], dtype=np.float32)#中清路line12
#ipm_pts = np.array([[430,587], [894,587], [894,694], [430,694]], dtype=np.float32)#中清路line12(new)
#ipm_pts = np.array([[1108,516], [1235,516], [1235,576], [1108,576]], dtype=np.float32)#中清路line34(new)
#ipm_pts = np.array([[562,711], [687,711], [687,825], [562,825]], dtype=np.float32)#台灣大道line12
ipm_pts = np.array([[1074,639], [1174,639], [1174,714], [1074,714]], dtype=np.float32)#台灣大道line12(new)
ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts) #轉置
inv_ipm_matrix = cv2.getPerspectiveTransform(ipm_pts, pts) #反轉置


ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])
#inv_ipm = cv2.warpPerspective(img2, inv_ipm_matrix, img2.shape[:2][::-1])



print(img.shape[:2][::-1])

# display (or save) images
cv2.imshow('img', img)
cv2.imshow('ipm', ipm) #轉置
#cv2.imshow('inv_ipm', inv_ipm) #反轉置
cv2.imwrite("C:\\Users\\ADMIN\\Desktop\\yolov4-deepsort\\outputs\\taichung_ipm_0424(3)(4).jpg",ipm)
#cv2.imwrite("taichung_inv_ipm_out.jpg",inv_ipm)
cv2.waitKey()

