"""Exports a pytorch *.pt model to *.onnx format

Usage:
    import torch
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import onnx
import warnings
warnings.filterwarnings("ignore")

from components.hand_detect.models.common import *
import sys
sys.path.append('components/hand_detect')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolo_weights/hand_s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    # google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights)['model']
    model.eval()
    model.fuse()

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(img)  # dry run
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['output'])  # output_names=['classes', 'boxes']

    # Check onnx model
    model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model)  # check onnx model
    print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
