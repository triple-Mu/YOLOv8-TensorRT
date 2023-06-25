from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import ALPHA, CLASSES, COLORS, MASK_COLORS
from models.torch_utils import seg_postprocess
from models.utils import blob, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['outputs', 'proto'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels, masks = seg_postprocess(
            data, bgr.shape[:2], args.conf_thres, args.iou_thres)
        masks = masks[:, dh:H - dh, dw:W - dw, :]
        mask_colors = MASK_COLORS[labels % len(MASK_COLORS)]
        mask_colors = mask_colors.reshape(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = cv2.resize(seg_img.astype(np.uint8), draw.shape[:2][::-1])

        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label, mask) in zip(bboxes, scores, labels, masks):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            mask = cv2.resize(mask, (w, h))
            mask = mask[dh: h - dh, dw: w - dw]
            bgr_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            bgr_mask[mask > 0.5] = MASK_COLORS[label % len(MASK_COLORS)]
            bgr_mask = cv2.resize(bgr_mask, (draw.shape[1], draw.shape[0]), cv2.INTER_NEAREST)
            bgr_mask[bgr_mask==0] = draw[bgr_mask==0]
            draw = cv2.addWeighted(draw, 0.5, bgr_mask, 0.5, 0.0)
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


def parse_args() -> argparse.Namespace:
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
