import os
import time

import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
# 加载模型组件库
from components.hand_detect.yolov5_hand import yolov5_hand_model
from components.hand_keypoints.handpose_x import handpose_x_model
from components.hand_track.hand_track import Tracker, pred_gesture
from components.hand_gesture.resnet import resnet18
from lib.hand_lib.utils.utils import parse_data_cfg

import torch
import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('components/hand_detect')


class RecogSystem(object):
    def __init__(self, args):
        self.args = args
        self.hand_detect_model = yolov5_hand_model(conf_thres=float(args.detect_conf_thres),
                                                   nms_thres=float(args.detect_nms_thres),
                                                   model_arch=args.detect_model_arch,
                                                   model_path=args.detect_model_path)
        # print(self.hand_detect_model)

        self.handpose_model = handpose_x_model(model_arch=args.handpose_x_model_arch,
                                               model_path=args.handpose_x_model_path)

        self.gesture_model = resnet18()
        self.gesture_model.load_state_dict(torch.load(args.gesture_model_path))
        self.gesture_model.eval()

    def __call__(self, img):
        print(img)#
        hand_bbox = self.hand_detect_model.predict(img, vis=True)
        if (hand_bbox is None) or len(hand_bbox) == 0:
            return None, None, None
        gesture_list = []
        for h_box in hand_bbox:
            x_min, y_min, x_max, y_max, score = h_box
            w_ = max(abs(x_max - x_min), abs(y_max - y_min))
            if w_ < 60:
                continue
            w_ = w_ * 1.26

            x_mid = (x_max + x_min) / 2
            y_mid = (y_max + y_min) / 2

            x1, y1, x2, y2 = int(x_mid - w_ / 2), int(y_mid - w_ / 2), int(x_mid + w_ / 2), int(y_mid + w_ / 2)

            x1 = np.clip(x1, 0, img.shape[1] - 1)
            x2 = np.clip(x2, 0, img.shape[1] - 1)

            y1 = np.clip(y1, 0, img.shape[0] - 1)
            y2 = np.clip(y2, 0, img.shape[0] - 1)

            box = [x1, y1, x2, y2]
            # box = [float(x1), float(y1), float(x2), float(y2)]

            pts_ = self.handpose_model.predict(img[y1:y2, x1:x2, :])  # 预测手指关键点

            gesture_name, score = pred_gesture(box, pts_, img, self.gesture_model)
            print(gesture_name)
            gesture_list.append([box, gesture_name, score])

            return box, gesture_name, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand Pose Inference')
    parser.add_argument('--cfg_file', type=str, default='lib/hand_lib/cfg/handpose.cfg', help='model_path')  # 模型路径

    parser.add_argument('--test_path', type=str,
                        default='/home/kls/hand/hand_pose/handpose_data/handpose_x_gesture_v1/000-one',
                        help='test_path')  # 测试图片路径 'weights/handpose_x_gesture_v1/handpose_x_gesture_v1/000-one' camera_id
    #
    parser.add_argument('--is_video', type=bool, default=False, help='if test_path is video')  # 是否视频
    parser.add_argument('--video_path', type=str, default='0', help='0 for cam/path ')  # 是否视频

    # parser.add_argument('--is_video', type=bool, default=True, help='if test_path is video')  # 是否视频
    # parser.add_argument('--video_path', type=str, default='video158.mp4', help='0 for cam/path ')  # 是否视频

    print('\n/******************* {} ******************/\n'.format(parser.description))
    args = parser.parse_args()  # 解析添加参数

    config = parse_data_cfg(args.cfg_file)
    is_videos = args.is_video
    test_path = args.test_path
    pred = RecogSystem(config)
    print(pred.predict(test_path))
