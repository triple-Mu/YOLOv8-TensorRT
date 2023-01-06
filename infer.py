from models import TRTModule, TRTProfilerV0
from pathlib import Path
import cv2
import argparse
import numpy as np
import torch
import random

random.seed(0)

SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm')
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush')

COLORS = {cls: [random.randint(0, 255) for _ in range(3)] for i, cls in enumerate(CLASSES)}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, np.array([r, r, r, r], dtype=np.float32), np.array([dw, dh, dw, dh], dtype=np.float32)


def blob(im):
    im = im.transpose(2, 0, 1)
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    return im


def main(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)

    images_path = Path(args.imgs)
    assert images_path.exists()
    save_path = Path(args.out_dir)

    if images_path.is_dir():
        images = [i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        ratio = torch.asarray(ratio, dtype=torch.float32, device=device)
        dwdh = torch.asarray(dwdh, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        num_dets, bboxes, scores, labels = Engine(tensor)
        bboxes = bboxes[0, :num_dets.item()]
        scores = scores[0, :num_dets.item()]
        labels = labels[0, :num_dets.item()]
        bboxes -= dwdh
        bboxes /= ratio
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw, f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [225, 255, 255], thickness=2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--out-dir', type=str, default='./output', help='Path to output file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='TensorRT infer device')
    parser.add_argument(
        '--profile', action='store_true', help='Profile TensorRT engine')
    args = parser.parse_args()
    return args


def profile(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    profiler = TRTProfilerV0()
    Engine.set_profiler(profiler)
    random_input = torch.randn(Engine.inp_info[0].shape, device=device)
    _ = Engine(random_input)


if __name__ == '__main__':
    args = parse_args()
    if args.profile:
        profile(args)
    else:
        main(args)
