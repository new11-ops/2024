img_path="/home/data/CZP/2024megc_test/test_crop/SAMM_data"
save_path="/home/data/CZP/2024megc_test/SAMM_align"
import os
import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector() #获取人脸分类器
predictor = dlib.shape_predictor('/home/data/CZP/MEGC/MEGC/spot/shape_predictor_68_face_landmarks.dat')    # 获取人脸检测器
# Dlib 检测器和预测器
font = cv2.FONT_HERSHEY_SIMPLEX

videos=os.listdir(img_path)
for video in videos:
    if not os.path.exists(os.path.join(save_path,video)):
        os.makedirs(os.path.join(save_path,video))
    print(video)
        
    imgs=os.listdir(os.path.join(img_path,video))
    imgs.sort()
    
    image_1 = cv2.imread(os.path.join(img_path,video,imgs[0]))
    img_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    faces = detector(image_1, 0)
    for i in range(len(faces)):
    #取特征点坐标
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image_1, faces[i]).parts()])
    left=landmarks[39]
    right=landmarks[42]
    nose = landmarks[33]
    src_pts = np.float32([[left[0,0],left[0,1]],[right[0,0],right[0,1]],[nose[0,0],nose[0,1]]])
    cv2.imwrite(os.path.join(save_path,video,imgs[0]),image_1)    
    for img in imgs[1:]:
        image = cv2.imread(os.path.join(img_path,video,img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray, 0)
        for i in range(len(faces)):
        #取特征点坐标
            landmarks = np.matrix([[p.x, p.y] for p in predictor(image, faces[i]).parts()])
        left_new=landmarks[39]
        right_new=landmarks[42]
        nose_new = landmarks[33]
        dst_pts = np.float32([[left_new[0,0],left_new[0,1]],[right_new[0,0],right_new[0,1]],[nose_new[0,0],nose_new[0,1]]])
        M= cv2.getAffineTransform(dst_pts,src_pts)
        warped = cv2.warpAffine(image, M, (image_1.shape[1], image_1.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        cv2.imwrite(os.path.join(save_path,video,img),warped)   
    
        
        