import os
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindface.detection.models import RetinaFace, resnet50, mobilenet025
from mindface.detection.runner import DetectionEngine
from mindface.detection.utils import prior_box

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
backbone = mobilenet025(1000)
network = RetinaFace(phase='predict', backbone=backbone)
backbone.set_train(False)
network.set_train(False)
param_dict = load_checkpoint('/home/data/CZP/MEGC/MEGC/ms_spot/models/RetinaFace_mobilenet025.ckpt')
load_param_into_net(network, param_dict)
detector = DetectionEngine()

path_root = '/home/data/CZP/MEGC/MEGC/data/test_set/SAMM'
path_save = '/home/data/CZP/MEGC/MEGC/data/test_set/SAMM_crop'
sub_path = os.listdir(path_root)

for sub in sub_path:
    print(sub)
    # if sub[:4] == 'SAMM':
    #     continue
    image_path = os.listdir(path_root+'/'+sub)
    if not os.path.exists(path_save+'/'+sub):
        os.makedirs(path_save+'/'+sub)
    # for imagepath in image_path:
        # print(sub+'/'+imagepath)
        # image_path = os.listdir(os.path.join(path_root+'/'+sub+'/'+imagepath))
        # if not os.path.exists(os.path.join(path_save+'/'+sub+'/'+imagepath)):
        #     os.mkdir(os.path.join(path_save+'/'+sub+'/'+imagepath))
    for image in image_path:
        target_size = 224
        max_size = 224
        priors = prior_box(image_sizes=(max_size, max_size),
                            min_sizes=[[16, 32], [64, 128], [256, 512]],
                            steps=[8, 16, 32],
                            clip=False)
        img = cv2.imread(os.path.join(path_root,sub,image), cv2.IMREAD_COLOR)
        # img = 
        img = np.float32(img)

        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t

        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = Tensor(img)

        boxes, confs, _ = network(img)
        boxes = detector.infer(boxes, confs, resize, scale, priors)
        
        img_crop = cv2.imread(os.path.join(path_root,sub,image), cv2.IMREAD_COLOR)
        if len(boxes) > 1 or boxes[0][4]<0.5:
            print(os.path.join(path_root,sub,image))
        box = boxes[0]
        box[1] = box[1]-50
        
        img_crop = img_crop[int(box[1])-100:int(box[1])+int(box[3])+100,int(box[0])-50:int(box[0])+int(box[2])+50]
        # img_crop = cv2.resize(img_crop, (320, 320))
        img_save = os.path.join(path_save+'/'+sub+'/'+image)
        # print(img_crop.shape)
        cv2.imwrite(img_save, img_crop)
        # print(boxes)
