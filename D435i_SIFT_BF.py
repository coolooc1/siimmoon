import cv2
import numpy as np
import pyrealsense2 as rs

# 对模板图像进行颜色空间转换为灰度图
img1 = cv2.imread("anjian1.jpg")
img2 = cv2.imread("anjian2.jpg")
img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

# 在循环外部计算SIFT特征点
sift = cv2.SIFT_create(nfeatures=1000)  
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 查找轮廓的中心
def find_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

paused = False

# 配置RealSense D435i
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 

# 启动RealSense pipeline
profile = pipeline.start(config)  # 将返回的对象分配给变量'profile'

# 获取深度传感器的深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 创建align对象，用于深度图像与彩色图像对齐
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        if not paused:
            # 从相机获取新帧
            frames = pipeline.wait_for_frames(10000)
             # 对齐深度图像和彩色图像
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() 

            img3 = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 将实时图像转换为灰度图像
            img3_gray = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)

            # 计算实时图像的SIFT特征点
            kp3, des3 = sift.detectAndCompute(img3_gray, None)

            # 如果没有找到任何一个图像的特征描述符，则跳过当前帧
            if des1 is None or des2 is None or des3 is None:
                cv2.imshow("img3", img3)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 使用暴力匹配器进行特征匹配
            bf = cv2.BFMatcher()
            matches1 = bf.knnMatch(des1, des3, k=2)
            matches2 = bf.knnMatch(des2, des3, k=2)

            # 选择较好的匹配
            good1 = [m[0] for m in matches1 if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
            good2 = [m[0] for m in matches2 if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

            # 获取匹配点的坐标
            src_pts1 = np.float32([kp1[m.queryIdx].pt for m in good1]).reshape(-1, 1, 2)
            src_pts2 = np.float32([kp2[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
            dst_pts1 = np.float32([kp3[m.trainIdx].pt for m in good1]).reshape(-1, 1, 2)
            dst_pts2 = np.float32([kp3[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)

            img3_combined = img3.copy()

            # 如果匹配点的数量不足4个，跳过当前帧
            if len(src_pts1) < 4 or len(dst_pts1) < 4 or len(src_pts2) < 4 or len(dst_pts2) < 4:
                cv2.imshow("img3_combined", img3_combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 使用RANSAC计算单应性矩阵
            if len(src_pts1) >= 4 and len(dst_pts1) >= 4:
                H1, _ = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
            else:
                H1 = None

            if len(src_pts2) >= 4 and len(dst_pts2) >= 4:
                H2, _ = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)
            else:
                H2 = None

            # 使用单应性矩阵对模板图像进行变换
            height, width, _ = img3.shape
            if H1 is not None:
                img1_warped = cv2.warpPerspective(img1_gray, H1, (width, height))
            else:
                img1_warped = img1_gray.copy()

            if H2 is not None:
                img2_warped = cv2.warpPerspective(img2_gray, H2, (width, height))
            else:
                img2_warped = img2_gray.copy()

            # 生成模板图像的二值化掩码
            ret, img1_mask = cv2.threshold(img1_warped, 1, 255, cv2.THRESH_BINARY)
            ret, img2_mask = cv2.threshold(img2_warped, 1, 255, cv2.THRESH_BINARY)

            # 确定二值化灰度模板图像的边缘
            contours1, _ = cv2.findContours(img1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(img2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 用原始相机图像初始化img3_combined
            img3_combined = img3.copy()
            if H1 is not None and H2 is not None:
                # 在img3_combined上绘制轮廓
                cv2.drawContours(img3_combined, contours1, -1, (0, 255, 0), 3)
                cv2.drawContours(img3_combined, contours2, -1, (0, 0, 255), 3)

                center1 = find_contour_center(contours1[0]) if contours1 else None
                center2 = find_contour_center(contours2[0]) if contours2 else None

                # 如果轮廓的中心点存在
                if center1 is not None and center2 is not None:
                    cX1, cY1 = center1
                    cX2, cY2 = center2

                    # 获取深度值（以毫米为单位）
                    depth1 = depth_image[cY1, cX1] * depth_scale * 1000
                    depth2 = depth_image[cY2, cX2] * depth_scale * 1000

                    # 在img3_combined上显示轮廓中心的坐标
                    text_offset_y = img3_combined.shape[0] - 20
                    cv2.putText(img3_combined, "center of VOR_LOC: ({}, {}, {:.2f}mm)".format(cX1, cY1, depth1), (img3_combined.shape[1] // 2 - 150, text_offset_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(img3_combined, "center of APP: ({}, {}, {:.2f}mm)".format(cX2, cY2, depth2), (img3_combined.shape[1] // 2 - 150, text_offset_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示图像
        cv2.imshow("img3_combined", img3_combined)

        # 获取用户按下的键
        key = cv2.waitKey(1) & 0xFF

        # 如果用户按下'q'键，退出循环
        if key == ord('q'):
            break

        # 如果用户按下空格键，暂停/取消暂停循环
        if key == ord(' '):
            paused = not paused

finally:
# 停止RealSense管道并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()     