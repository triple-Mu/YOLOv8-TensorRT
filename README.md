# YOLOv8-TensorRT
YOLOv8 using TensorRT accelerate !

# Preprocessed ONNX model
You can dowload the onnx model which is pretrained by https://github.com/ultralytics .

[**YOLOv8-n**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8n_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1772936700&Signature=r6HgJTTcCSAxQxD9bKO9qBTtigQ%3D)

[**YOLOv8-s**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8s_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936722&Signature=JjxQFx1YElcVdsCaMoj81KJ4a5s%3D)

[**YOLOv8-m**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8m_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936739&Signature=IRKBELdVFemD7diixxxgzMYqsWg%3D)

[**YOLOv8-l**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8l_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1682936763&Signature=RGkJ4G2XJ4J%2BNiki5cJi3oBkDnA%3D)

[**YOLOv8-x**](https://triplemu.oss-cn-beijing.aliyuncs.com/YOLOv8/ONNX/yolov8x_nms.onnx?OSSAccessKeyId=LTAI5tN1dgmZD4PF8AJUXp3J&Expires=1673936778&Signature=3o%2F7QKhiZg1dW3I6sDrY4ug6MQU%3D)

# Build TensorRT engine by ONNX

## 1. By TensorRT Python api

You can export TensorRT engine by [`build.py` ](build.py).

Usage: 

``` shell
python3 build.py --onnx yolov8s_nms.onnx --device cuda:0 --fp16
```

#### Description of all arguments

- `--onnx` : The ONNX model you download.
- `--device` : The CUDA deivce you export engine .
- `--half` : Whether to export half-precision model.

## 2. By trtexec tools

You can export TensorRT engine by [`trtexec`](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) tools.

Usage:

``` shell
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s_nms.onnx --saveEngine=yolov8s_nms.engine --fp16
```

***If you installed TensorRT by a debian package, then the installation path of `trtexec` is `/usr/src/tensorrt/bin/trtexec`***

***If you installed TensorRT by a tar package, then the installation path of trtexec is under the `bin` folder in the path you decompressed***

# Infer images by the engine which you export

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

  

