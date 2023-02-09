from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from infer import SUFFIXS, CLASSES, COLORS, MASK_COLORS, ALPHA
from infer import letterbox, blob, crop_mask, profile


def seg_postprocess(
        data: Tuple[Tensor, Tensor, Tensor, Tensor],
        shape: Union[Tuple, List]) \
        -> Tuple[Tensor, Tensor, Tensor, List]:
    assert len(data) == 4
    bboxes, scores, labels, masks = data
    idx = scores > 0
    bboxes, scores, labels, masks = bboxes[idx], scores[idx], labels[
        idx], masks[idx]
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(
        masks[None], shape, mode='bilinear', align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def get_mask_color(labels, masks):
    cidx = (labels % len(MASK_COLORS)).cpu().numpy()
    mask_color = torch.tensor(MASK_COLORS[cidx].reshape(-1, 1, 1,
                                                        3)).to(masks) * ALPHA
    return masks @ mask_color


def main(args):
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['bboxes', 'scores', 'labels', 'masks'])

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

        seg_img = torch.asarray(
            seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]], device=device)
        bboxes, scores, labels, masks = seg_postprocess(data, bgr.shape[:2])
        mask_color = get_mask_color(labels, masks)
        mask, mask_color = [
            m[:, dh:H - dh, dw:W - dw, :] for m in [masks, mask_color]
        ]
        inv_alph_masks = (1 - mask * 0.5).cumprod(0)
        mcs = (mask_color * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                          draw.shape[:2][::-1])

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Profile TensorRT engine')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.profile:
        profile(args)
    else:
        main(args)
