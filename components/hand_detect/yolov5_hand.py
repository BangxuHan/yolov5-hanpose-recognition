import os
import random
import time

import cv2
import numpy as np
import torch
import torchvision

from components.hand_detect.utils.utils import *
from components.hand_detect.models.yolo import Model


def process_data(img, img_size=640):  # 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RG25
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    # crop_img = img.copy()
    # x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    # crop = crop_img[y1:y2, x1:x2]
    # t = time.time()
    # t = int(round(t * 1000))
    # cv2.imwrite(f'hand/{t}.jpg', crop)

    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 55, 90], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imwrite('img.jpg', img)


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img_size, coords, img0_shape):  # image size 转为 原图尺寸
    # Rescale x1, y1, x2, y2 from 416 to image size
    # print('coords     : ',coords)
    # print('img0_shape : ',img0_shape)
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    # print('gain       : ',gain)
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    # print('pad_xpad_y : ',pad_x,pad_y)
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)  # 夹紧区间最小值不为负数
    return coords


def letterbox(img, height=640, augment=False, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resize img
    if augment:
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # print("resize time:",time.time()-s1)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


class yolov5_hand_model(object):
    def __init__(self,
                 model_path='weights/yolo_weights/hand_s.pt',
                 model_arch='components/hand_detect/models/yolov5s.yaml',
                 img_size=640,
                 conf_thres=0.16,
                 nms_thres=0.4):
        print("yolov5 hand_model loading: {}".format(model_path))
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.img_size = img_size
        self.classes = ["Hand"]
        self.num_classes = len(self.classes)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # -----------------------------------------------------------------------
        weights = model_path
        # if "yolov5" in model_arch:
        #     model = Model().to(self.device)
        # -----------------------------------------------------------------------

        self.model = Model(model_arch).to(self.device)
        # print(self.model)  # 显示模型参数

        # print('num_classes : ', self.num_classes)

        # Load weights
        if os.access(weights, os.F_OK):  # 判断模型文件是否存在
            # state_dict = torch.load(weights)['model']
            # print(state_dict)
            # self.model.state_dict()
            # self.model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage)['model'])
            self.model = torch.load(weights, map_location=self.device)['model']
        else:
            print('------- >>> error : model not exists')

        self.model.eval()  # 模型设置为 eval
        # acc_model('', self.model)
        # self.model = self.model.to(self.device)

    def predict(self, img_, vis):
        with torch.no_grad():
            t = time.time()
            img = process_data(img_, self.img_size)
            t1 = time.time()
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            pred = self.model(img)  # 图片检测

            t2 = time.time()
            detections = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]  # nms
            t3 = time.time()
            # print("t3 time:", t3)

            if (detections is None) or len(detections) == 0:
                return []
            # Rescale boxes from 416 to true image size
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img_.shape).round()
            # Show detect reslut
            dets_for_landmarks = []
            colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, 10 + 1)][::-1]

            output_dict_ = []
            for *xyxy, conf, cls in detections:
                label = '%s %.2f' % (self.classes[0], conf)
                x1, y1, x2, y2 = xyxy
                output_dict_.append((float(x1), float(y1), float(x2), float(y2), float(conf.item())))
                if vis:
                    plot_one_box(xyxy, img_, label=label, color=(0, 175, 255), line_thickness=2)
            return output_dict_


if __name__ == '__main__':
    a = yolov5_hand_model()
    print(a)
