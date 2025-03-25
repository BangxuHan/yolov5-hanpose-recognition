import base64
import json
import cv2
import numpy as np
import copy
import time
import torch
from flask import Flask, request
from flask_restful import Resource, Api

import threading
import argparse
app = Flask("HandposeRecog")
api = Api(app)


_infer_lock = threading.Lock()

import predict_system

import sys
sys.path.append('components/hand_detect')


class HandRecognition(object):
    def __init__(self, args):
        self.args = args
        self.model = predict_system.RecogSystem(args)

    def __call__(self, img):
        print("start handpose recognition process")
        box, gesture, score = self.model.__call__(img)
        return box, gesture, score

    def predict(self, imgs):
        res = []
        for idx, img in enumerate(imgs):
            gesture_box, gesture_name, gesture_score = self.__call__(img)
            if gesture_name is not None:
                res.append([gesture_box, gesture_name, gesture_score])
            else:
                return [None]
        return res


class HandArgs():
    use_gpu = True
    detect_model_path = 'weights/yolo_weights/hand_m.pt'
    detect_model_arch = 'components/hand_detect/models/yolov5m.yaml'
    detect_conf_thres = 0.31
    detect_nms_thres = 0.45

    handpose_x_model_path = 'weights/handpose_x_weights/squeezenet1_1-size-256-loss-0.0732.pth'
    handpose_x_model_arch = 'squeezenet1_1'
    gesture_model_path = 'weights/resnet18_classify/resnet18-300-regular.pth'

    def __init__(self):
        pass


class PoseRecognition(Resource):
    def post(self):
        temp = request.get_data(as_text=True)
        data = json.loads(temp)
        images = data['image']
        imagebuf = []
        for imagestr in images:
            imagedata_base64 = base64.b64decode(imagestr)
            np_arr = np.frombuffer(imagedata_base64, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            imagebuf.append(image)

        with _infer_lock:
            det_res = model.predict(imagebuf)
            print(det_res)

        words_res = []
        nlen = len(det_res)

        for i in range(nlen):
            if det_res[i] is not None:
                print('-----------', det_res[i][0])
                float_res = []
                for num in det_res[i][0]:
                    float_res.append(float(num))
                # float_res = [float(num) for num in det_res[i][0] if isinstance(num, (int, float))]
                print(float_res)
                temp = {
                    "hand_box": float_res,
                    "hand_pose": det_res[i][1],
                    "pose_conf": det_res[i][2].__float__()
                }
                words_res.append(temp)
            else:
                nlen = 0

        result = {"result_number": nlen,
                  "result_gesture": words_res
                  }
        return app.response_class(json.dumps(result), mimetype='application/json')


api.add_resource(PoseRecognition, '/poserecog')
args = HandArgs()
model = HandRecognition(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='handpose recognition server port')
    args = parser.parse_args()
    port = args.port
    app.run(host='0.0.0.0', port=port, debug=False)
