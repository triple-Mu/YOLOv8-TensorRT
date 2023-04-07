# YOLOv8-pose Model with TensorRT

The yolov8-pose model conversion route is :
    YOLOv8 PyTorch model -> ONNX -> TensorRT Engine

***Notice !!!*** We don't support TensorRT API building !!!

# Export Orin ONNX model by ultralytics

You can leave this repo and use the original `ultralytics` repo for onnx export.

### 1. Python script

Usage:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="engine", device=0)  # export the model to engine format
assert success
```

After executing the above script, you will get an engine named `yolov8s-pose.engine` .

### 2. CLI tools

Usage:

```shell
yolo export model=yolov8s-pose.pt format=engine device=0
```

After executing the above command, you will get an engine named `yolov8s-pose.engine` too.

## Inference with c++

You can infer with c++ in [`csrc/pose/normal`](../csrc/pose/normal) .

### Build:

Please set you own librarys in [`CMakeLists.txt`](../csrc/pose/normal/CMakeLists.txt) and modify `KPS_COLORS` and `SKELETON` and  `LIMB_COLORS`  in [`main.cpp`](../csrc/pose/normal/main.cpp).

Besides, you can modify the postprocess parameters such as `score_thres` and `iou_thres` and `topk` in [`main.cpp`](../csrc/pose/normal/main.cpp).

```c++
int topk = 100;
float score_thres = 0.25f;
float iou_thres = 0.65f;
```

And build:

``` shell
export root=${PWD}
cd src/pose/normal
mkdir build
cmake ..
make
mv yolov8-pose ${root}
cd ${root}
```

Usage:

``` shell
# infer image
./yolov8-pose yolov8s-pose.engine data/bus.jpg
# infer images
./yolov8-pose yolov8s-pose.engine data
# infer video
./yolov8-pose yolov8s-pose.engine data/test.mp4 # the video path
```
