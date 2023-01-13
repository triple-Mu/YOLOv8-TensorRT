# YOLOv8-TensorRT

`YOLOv8` using TensorRT accelerate !

---
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![Python Version](https://img.shields.io/badge/Python-3.8--3.10-FFD43B?logo=python)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![img](https://badgen.net/badge/icon/tensorrt?icon=azurepipelines&label)](https://developer.nvidia.com/tensorrt)
[![C++](https://img.shields.io/badge/CPP-11%2F14-yellow)](https://github.com/triple-Mu/YOLOv8-TensorRT)
[![img](https://badgen.net/github/license/triple-Mu/YOLOv8-TensorRT)](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/LICENSE)
[![img](https://badgen.net/github/prs/triple-Mu/YOLOv8-TensorRT)](https://github.com/triple-Mu/YOLOv8-TensorRT/pulls)
[![img](https://img.shields.io/github/stars/triple-Mu/YOLOv8-TensorRT?style=social&label=Star&maxAge=2592000)](https://github.com/triple-Mu/YOLOv8-TensorRT)

---


# Prepare the environment

1. Install TensorRT follow [`TensorRT offical website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

2. Install python requirement.

   ``` shell
   pip install -r requirement.txt
   ```

3. (optional) Install [`ultralytics`](https://github.com/ultralytics/ultralytics) package for ONNX export or TensorRT API building.

   ``` shell
   pip install ultralytics
   ```

   You can download pretrained pytorch model by:

   ``` shell
   wget https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt
   wget https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt
   wget https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8m.pt
   wget https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8l.pt
   wget https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8x.pt
   ```

# Build TensorRT Engine by ONNX


## Export ONNX by `ultralytics` API

### Export Your Own ONNX model

You can export your onnx model by `ultralytics` API
and add postprocess into model at the same time.

``` shell
python3 export.py \
--weights yolov8s.pt \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--opset 11 \
--sim \
--input-shape 1 3 640 640 \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The PyTorch model you trained.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--opset` : ONNX opset version, default is 11.
- `--sim` : Whether to simplify your onnx model.
- `--input-shape` : Input shape for you model, should be 4 dimensions.
- `--device` : The CUDA deivce you export engine .

You will get an onnx model whose prefix is the same as input weights.

###  Just Taste First

If you just want to taste first, you can download the onnx model which are exported by `YOLOv8` package and modified by me.

[**YOLOv8-n**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8n_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1772936700&Signature=r6HgJTTcCSAxQxD9bKO9qBTtigQ%3D)

[**YOLOv8-s**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8s_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936722&Signature=JjxQFx1YElcVdsCaMoj81KJ4a5s%3D)

[**YOLOv8-m**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8m_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936739&Signature=IRKBELdVFemD7diixxxgzMYqsWg%3D)

[**YOLOv8-l**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8l_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936763&Signature=RGkJ4G2XJ4J%2BNiki5cJi3oBkDnA%3D)

[**YOLOv8-x**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8x_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1673936778&Signature=3o%2F7QKhiZg1dW3I6sDrY4ug6MQU%3D)

## Export Engine by TensorRT Python api

You can export TensorRT engine from ONNX by [`build.py` ](build.py).

Usage:

``` shell
python3 build.py \
--weights yolov8s.onnx \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--device cuda:0
```

#### Description of all arguments

- `--weights` : The ONNX model you download.
- `--iou-thres` : IOU threshold for NMS plugin.
- `--conf-thres` : Confidence threshold for NMS plugin.
- `--topk` : Max number of detection bboxes.
- `--fp16` : Whether to export half-precision engine.
- `--device` : The CUDA deivce you export engine .

You can modify `iou-thres` `conf-thres` `topk` by yourself.

## 2. Export Engine by Trtexec Tools

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

``` shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolov8s.onnx \
--saveEngine=yolov8s.engine \
--fp16
```

**If you installed TensorRT by a debian package, then the installation path of `trtexec`
is `/usr/src/tensorrt/bin/trtexec`**

**If you installed TensorRT by a tar package, then the installation path of `trtexec` is under the `bin` folder in the path you decompressed**

# Build TensorRT Engine by TensorRT API

Please see more information in [`API-Build.md`](docs/API-Build.md)

***Notice !!!*** We don't support YOLOv8-seg model now !!!

# Inference

## 1. Infer with python script

You can infer images with the engine by [`infer.py`](infer.py) .

Usage:

``` shell
python3 infer.py \
--engine yolov8s.engine \
--imgs data \
--show \
--out-dir outputs \
--device cuda:0
```

#### Description of all arguments

- `--engine` : The Engine you export.
- `--imgs` : The images path you want to detect.
- `--show` : Whether to show detection results.
- `--out-dir` : Where to save detection results images. It will not work when use `--show` flag.
- `--device` : The CUDA deivce you use.
- `--profile` : Profile the TensorRT engine.

## 2. Infer with C++

You can infer with c++ in [`csrc/detect`](csrc/detect) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](csrc/detect/CMakeLists.txt) and modify you own config in [`config.h`](csrc/detect/include/config.h) such as `CLASS_NAMES` and `COLORS`.

``` shell
export root=${PWD}
cd src/end2end
mkdir build
cmake ..
make
mv yolov8 ${root}
cd ${root}
```

Usage:

``` shell
# infer image
./yolov8 yolov8s.engine data/bus.jpg
# infer images
./yolov8 yolov8s.engine data
# infer video
./yolov8 yolov8s.engine data/test.mp4 # the video path
```

# TensorRT Segment Deploy

Please see more information in [`Segment.md`](docs/Segment.md)

# DeepStream Detection Deploy

See more in [`README.md`](csrc/deepstream/README.md)


# Profile you engine

If you want to profile the TensorRT engine:

Usage:

``` shell
python3 infer.py --engine yolov8s.engine --profile
```
