# YOLOv8-TensorRT

`YOLOv8` using TensorRT accelerate !

# Prepare the environment

1. Install TensorRT follow [`TensorRT offical website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

2. Install python requirement.

   ``` shell
   pip install -r requirement.txt
   ```

3. (optional) Install `ultralytics YOLOv8` package for TensorRT API building.

   ``` shell
   pip install -i https://test.pypi.org/simple/ ultralytics
   ```

   You can download pretrained pytorch model by:

   ``` shell
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   ```

# Build TensorRT engine by ONNX

## Preprocessed ONNX model

You can dowload the onnx model which are exported by `YOLOv8` package and modified by me  .

[**YOLOv8-n**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8n_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1772936700&Signature=r6HgJTTcCSAxQxD9bKO9qBTtigQ%3D)

[**YOLOv8-s**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8s_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936722&Signature=JjxQFx1YElcVdsCaMoj81KJ4a5s%3D)

[**YOLOv8-m**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8m_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936739&Signature=IRKBELdVFemD7diixxxgzMYqsWg%3D)

[**YOLOv8-l**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8l_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936763&Signature=RGkJ4G2XJ4J%2BNiki5cJi3oBkDnA%3D)

[**YOLOv8-x**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8x_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1673936778&Signature=3o%2F7QKhiZg1dW3I6sDrY4ug6MQU%3D)

## 1. By TensorRT ONNX Python api

You can export TensorRT engine from ONNX by [`build.py` ](build.py).

Usage:

``` shell
python build.py \
--weights yolov8s_nms.onnx \
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

## 2. By trtexec tools

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

``` shell
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s_nms.onnx --saveEngine=yolov8s_nms.engine --fp16
```

**If you installed TensorRT by a debian package, then the installation path of `trtexec`
is `/usr/src/tensorrt/bin/trtexec`**

**If you installed TensorRT by a tar package, then the installation path of `trtexec` is under the `bin` folder in the path you decompressed**



# Build TensorRT engine by API

When you want to build engine by api. You should generate the pickle weights parameters first.

``` shell
python gen_pkl.py -w yolov8s.pt -o yolov8s.pkl
```

You will get a `yolov8s.pkl` which contain the operators' parameters. And you can rebuild `yolov8s` model in TensorRT api.

```
python build.py \
--weights yolov8s.pkl \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--input-shape 1 3 640 640 \
--device cuda:0
```

***Notice !!!***  Now we only support static input shape model build by TensorRT api. You'd best give the legal`input-shape`.

# Infer images by the engine which you export or build

## 1. Python infer

You can infer images with the engine by [`infer.py`](infer.py) .

Usage:

``` shell
python3 infer.py --engine yolov8s_nms.engine --imgs data --show --out-dir outputs --device cuda:0
```

#### Description of all arguments

- `--engine` : The Engine you export.
- `--imgs` : The images path you want to detect.
- `--show` : Whether to show detection results.
- `--out-dir` : Where to save detection results images. It will not work when use `--show` flag.
- `--device` : The CUDA deivce you use.
- `--profile` : Profile the TensorRT engine.

## 2. C++ infer

You can infer with c++ in [`csrc/end2end`](csrc/end2end) .

Build:

Please set you own librarys in [`CMakeLists.txt`](csrc/end2end/CMakeLists.txt) and modify you own config in [`yolov8.hpp`](csrc/end2end/yolov8.hpp) such as `classes names` and `colors` .

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
./yolov8 yolov8s_nms.engine data/bus.jpg
# infer images
./yolov8 yolov8s_nms.engine data
# infer video
./yolov8 yolov8s_nms.engine data/test.mp4 # the video path
```

# Profile you engine

If you want to profile the TensorRT engine:

Usage:

``` shell
python3 infer.py --engine yolov8s_nms.engine --profile
```
