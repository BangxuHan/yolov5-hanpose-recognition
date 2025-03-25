import torch
import tensorrt as trt
from components.hand_detect.models.yolo import Model
from components.hand_keypoints.models.rexnetv1 import ReXNetV1
from components.hand_gesture.resnet import resnet18
from components.hand_keypoints.models.squeezenet import squeezenet1_1
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('components/hand_detect')
# import config_file.common as common
import os
import cv2
import numpy as np

device = 'cpu'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def yolov5_model(torch_path):
    yolov5_model = Model('components/hand_detect/models/yolov5s.yaml')
    yolov5_model = torch.load(torch_path, map_location=device)['model']
    yolov5_model.to(device)
    return yolov5_model.eval()


def handpose_x_model(torch_path):
    # handpose_x_model = ReXNetV1(num_classes=42)
    handpose_x_model = squeezenet1_1(pretrained=True, num_classes=42)
    # handpose_x_model.set_swish(memory_efficient=False)
    handpose_x_model.to(device)
    chkpt = torch.load(torch_path, map_location=device)
    handpose_x_model.load_state_dict(chkpt)
    return handpose_x_model.eval()


def gesture_model(torch_path):
    gesture_model = resnet18()
    gesture_model.load_state_dict(torch.load(torch_path))
    gesture_model.to(device)
    return gesture_model.eval()


def torch_to_onnx(model_path, save_path, name, input_shape):
    if name == 'yolov5':
        torch_model = yolov5_model(model_path)
        dummy_input = torch.randn(input_shape).to(device)
        # dummy_input.to(torch.float64)
        input_names = ["input"]
        output_names = ["output"]
    elif name == 'handpose':
        torch_model = handpose_x_model(model_path)
        dummy_input = torch.randn(input_shape).to(device)
        input_names = ["input"]
        output_names = ["output"]
    elif name == 'gesture':
        torch_model = gesture_model(model_path)
        dummy_input = torch.randn(input_shape).to(device)
        input_names = ["input"]
        output_names = ["output"]
    print('Converting torch --> onnx')
    torch.onnx.export(torch_model.cpu(),
                      dummy_input.cpu(),
                      save_path,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)
    print('ONNX model convert successful, saved in {}'.format(save_path))
    print('=' * 100)


def GiB(val):
    return val * 1 << 30


def onnx_to_trt(input_size, onnx_file_path="", engine_file_path="", save_engine=True):
    print(input_size)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            builder.create_builder_config() as config:
        config.max_workspace_size = GiB(1)
        builder.max_batch_size = 1
        # config.set_flag(trt.BuilderFlag.FP16)
        if not os.path.exists(onnx_file_path):
            quit("未找到{}".format(onnx_file_path))
        else:
            print('加载onnx模型从{} ......'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:  # 二值化的网络结果和参数
                print("开始解析onnx文件....")
                parser.parse(model.read())  # 解析onnx文件
                print("完成onnx模型解析!")
                print("从文件'{}'构建一个引擎可能需要一些时间…".format(onnx_file_path))
                print(network.get_layer(network.num_layers - 1).get_output(0).shape)
                network.get_input(0).shape = input_size
                engine = builder.build_serialized_network(network, config)
                print("完成创建引擎")
                if save_engine:  # 保存engine供以后直接反序列化使用
                    with open(engine_file_path, 'wb') as f11:
                        f11.write(engine)  # 序列化
                return engine


'''
yolov5_config:
            batch_size = 1  
            channel = 3
            input_shape = [640, 640]

handpose_config:
                batch_size = 1  
                channel = 3
                input_shape = [256, 256]

gesture_config:
                batch_size = 1 
                channel = 3 
                input_shape = [128, 128]
'''


if __name__ == '__main__':
    # name = 'yolov5'
    name = 'handpose'
    # name = 'gesture'

    batch_size = 1
    channel = 3
    # input_shape = [640, 640]
    input_shape = [256, 256]
    # input_shape = [128, 128]

    if name == 'yolov5':
        input_shape = [batch_size, channel, input_shape[0], input_shape[1]]
        model_path = r'weights/yolo_weights/hand_s.pt'
        save_path = r'onnx_model/yolov5s_640.onnx'
        trt_path = r'trt_engine/yolov5s_640.trt'
        torch_to_onnx(model_path, save_path, name, input_shape)
        onnx_to_trt(input_size=input_shape, onnx_file_path=save_path, engine_file_path=trt_path)
    if name == 'handpose':
        input_shape = [batch_size, channel, input_shape[0], input_shape[1]]
        # model_path = r'weights/handpose_x_weights/resnet_50-size-256-wingloss102-0.119.pth'
        # save_path = r'onnx_model/resnet50_256.onnx'
        # trt_path = r'trt_engine/resnet50_256.trt'
        model_path = r'weights/handpose_x_weights/squeezenet1_1-size-256-loss-0.0732.pth'
        save_path = r'onnx_model/squeezenet1_1_256.onnx'
        trt_path = r'trt_engine/squeezenet1_1_256.trt'
        torch_to_onnx(model_path, save_path, name, input_shape)
        # onnx_to_trt(input_size=input_shape, onnx_file_path=save_path, engine_file_path=trt_path)
    if name == 'gesture':
        input_shape = [batch_size, channel, input_shape[0], input_shape[1]]
        model_path = r'weights/resnet18_classify/resnet18-300-regular.pth'
        save_path = r'onnx_model/resnet18_128.onnx'
        trt_path = r'trt_engine/resnet18_128.trt'
        torch_to_onnx(model_path, save_path, name, input_shape)
        onnx_to_trt(input_size=input_shape, onnx_file_path=save_path, engine_file_path=trt_path)


