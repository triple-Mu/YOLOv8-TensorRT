from models import TRTModule, TRTProfilerV0  # isort:skip
import argparse
import os
import random
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torchvision.ops import batched_nms

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

random.seed(0)

SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# the same as yolov8
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

ALPHA = 0.5


def letterbox(
    im: ndarray,
    new_shape: Union[Tuple, List] = (640, 640),
    color: Union[Tuple, List] = (114, 114, 114)
) -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray) -> Tuple[ndarray, ndarray]:
    seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    return im, seg


def main(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    if args.seg:
        Engine.set_desired(['outputs', 'proto'])
    else:
        Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images_path = Path(args.imgs)
    assert images_path.exists()
    save_path = Path(args.out_dir)

    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, seg_img = blob(rgb)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        data = Engine(tensor)

        if args.seg:
            seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]],
                                    device=device)
            bboxes, scores, labels, masks = seg_postprocess(
                data, bgr.shape[:2], args.conf_thres, args.iou_thres)
            mask, mask_color = [m[:, dh:H - dh, dw:W - dw, :] for m in masks]
            inv_alph_masks = (1 - mask * 0.5).cumprod(0)
            mcs = (mask_color * inv_alph_masks).sum(0) * 2
            seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
            draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                              draw.shape[:2][::-1])
        else:
            bboxes, scores, labels, masks = det_postprocess(data)

        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) -> Tuple[Tensor, Tensor, Tensor, List]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    select = scores > conf_thres
    bboxes, scores, labels, maskconf = bboxes[select], scores[select], labels[
        select], maskconf[select]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = bboxes[idx], scores[idx], labels[
        idx].int(), maskconf[idx]
    masks = (maskconf @ proto).view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    cidx = (labels % len(MASK_COLORS)).cpu().numpy()
    mask_color = torch.tensor(MASK_COLORS[cidx].reshape(-1, 1, 1,
                                                        3)).to(bboxes) * ALPHA
    out = [masks, masks @ mask_color]
    return bboxes, scores, labels, out


def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Any], **kwargs):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--seg', action='store_true', help='Seg inference')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='Confidence threshold')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Profile TensorRT engine')
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
