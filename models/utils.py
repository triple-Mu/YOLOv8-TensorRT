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
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def bbox_iou(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
                  (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
                  (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def batched_nms(boxes: ndarray,
                scores: ndarray,
                iou_thres: float = 0.65,
                conf_thres: float = 0.25):
    labels = np.argmax(scores, axis=-1)
    scores = np.max(scores, axis=-1)

    cand = scores > conf_thres
    boxes = boxes[cand]
    scores = scores[cand]
    labels = labels[cand]

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        while cls_boxes.shape[0] > 0:
            max_idx = np.argmax(cls_scores)
            max_box = cls_boxes[max_idx:max_idx + 1]
            max_score = cls_scores[max_idx:max_idx + 1]
            max_label = np.array([cls], dtype=np.int32)
            keep_boxes.append(max_box)
            keep_scores.append(max_score)
            keep_labels.append(max_label)
            other_boxes = np.delete(cls_boxes, max_idx, axis=0)
            other_scores = np.delete(cls_scores, max_idx, axis=0)
            ious = bbox_iou(max_box, other_boxes)
            iou_mask = ious < iou_thres
            if not iou_mask.any():
                break
            cls_boxes = other_boxes[iou_mask]
            cls_scores = other_scores[iou_mask]

    if len(keep_boxes) == 0:
        keep_boxes = np.empty((0, 4), dtype=np.float32)
        keep_scores = np.empty((0, ), dtype=np.float32)
        keep_labels = np.empty((0, ), dtype=np.float32)

    else:
        keep_boxes = np.concatenate(keep_boxes, axis=0)
        keep_scores = np.concatenate(keep_scores, axis=0)
        keep_labels = np.concatenate(keep_labels, axis=0)

    return keep_boxes, keep_scores, keep_labels


def nms(boxes: ndarray,
        scores: ndarray,
        iou_thres: float = 0.65,
        conf_thres: float = 0.25):
    labels = np.argmax(scores, axis=-1)
    scores = np.max(scores, axis=-1)

    cand = scores > conf_thres
    boxes = boxes[cand]
    scores = scores[cand]
    labels = labels[cand]

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    idxs = scores.argsort()
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_box = boxes[max_score_index:max_score_index + 1]
        max_score = scores[max_score_index:max_score_index + 1]
        max_label = np.array([labels[max_score_index]], dtype=np.int32)
        keep_boxes.append(max_box)
        keep_scores.append(max_score)
        keep_labels.append(max_label)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = bbox_iou(max_box, other_boxes)
        iou_mask = ious < iou_thres
        idxs = idxs[iou_mask]

    if len(keep_boxes) == 0:
        keep_boxes = np.empty((0, 4), dtype=np.float32)
        keep_scores = np.empty((0, ), dtype=np.float32)
        keep_labels = np.empty((0, ), dtype=np.float32)

    else:
        keep_boxes = np.concatenate(keep_boxes, axis=0)
        keep_scores = np.concatenate(keep_scores, axis=0)
        keep_labels = np.concatenate(keep_labels, axis=0)

    return keep_boxes, keep_scores, keep_labels


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
    iou_thres: float = 0.65
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    if nums == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(
            (0, ), dtype=np.float32), np.empty((0, ), dtype=np.int32)
    # check score negative
    scores[scores < 0] = 1 + scores[scores < 0]
    # add nms
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]

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
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), \
            np.empty((0,), dtype=np.float32), \
            np.empty((0,), dtype=np.int32), \
            np.empty((0, 0, 0, 0), dtype=np.int32)

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


def pose_postprocess(
        data: Union[Tuple, ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    bboxes, scores, kpts = np.split(outputs, [4, 5], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), np.empty(
            (0, ), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = np.split(bboxes, [
        2,
    ], -1)
    cvbboxes = np.concatenate([xycenter - 0.5 * wh, wh], -1)
    idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    cvbboxes, scores, kpts = cvbboxes[idx], scores[idx], kpts[idx]
    cvbboxes[:, 2:] += cvbboxes[:, :2]
    return cvbboxes, scores, kpts.reshape(idx.shape[0], -1, 3)
