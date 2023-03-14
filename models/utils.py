from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray

# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
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


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images


def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def det_postprocess(data: Tuple[ndarray, ndarray, ndarray, ndarray]):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels


def seg_postprocess(
        data: Tuple[ndarray],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]],
                              1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split('.')[:2])
    assert v0 == 4, 'OpenCV version is wrong'
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres,
                                      iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = masks.transpose([1, 2, 0])
    masks = cv2.resize(masks, (shape[1], shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    masks = masks.transpose(2, 0, 1)
    masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
    return bboxes, scores, labels, masks
