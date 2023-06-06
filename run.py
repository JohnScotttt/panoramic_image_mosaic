import cv2
import numpy as np

# 加载图像并转换为灰度图像
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 初始化SIFT特征检测器
sift = cv2.SIFT_create()

# 使用SIFT检测关键点和描述符
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 使用FLANN匹配器进行特征匹配
matcher = cv2.FlannBasedMatcher()
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

# 应用SRNSAC算法筛选最佳匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 提取关键点对应的位置
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法计算投影映射矩阵
homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

# 对图像2进行透视变换
result = cv2.warpPerspective(image2, homography, (image1.shape[1] + image2.shape[1], image2.shape[0]))
result[0:image1.shape[0], 0:image1.shape[1]] = image1

# 输出拼接结果
cv2.imwrite('Panorama.jpg', result)

