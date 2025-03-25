import torch
import warnings
warnings.filterwarnings("ignore")
# import sys
# sys.path.insert(0, 'components/hand_detect/models')
device = 'cuda:0'
weights = '/home/kls/PycharmProjects/yolov5-handpose-recognition/weights/yolo_weights/hand_s.pt'

# model = torch.load(weights, map_location=device)['model']
# print(model)
state_dict = torch.load(weights)['model']
print(state_dict)