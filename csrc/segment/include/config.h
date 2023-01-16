//
// Created by ubuntu on 1/16/23.
//

#ifndef YOLOV8_TENSORRT_CSRC_SEGMENT_INCLUDE_CONFIG_H
#define YOLOV8_TENSORRT_CSRC_SEGMENT_INCLUDE_CONFIG_H
#include "opencv2/opencv.hpp"
namespace seg
{
	const int DEVICE = 0;

	const int INPUT_W = 640;
	const int INPUT_H = 640;
	const int NUM_INPUT = 1;
	const int NUM_OUTPUT = 2;
	const int NUM_PROPOSAL = 8400; // feature map 20*20+40*40+80*80
	const int NUM_SEG_C = 32; // seg channel
	const int NUM_COLS = 6 + NUM_SEG_C; // x0 y0 x1 y1 score label 32

	const int SEG_W = 160;
	const int SEG_H = 160;

	// thresholds
	const float CONF_THRES = 0.25;
	const float IOU_THRES = 0.65;
	const float MASK_THRES = 0.5;

	// distance
	const float DIS = 7680.f;

	const int NUM_BINDINGS = NUM_INPUT + NUM_OUTPUT;
	const cv::Scalar PAD_COLOR = { 114, 114, 114 };
	const cv::Scalar RECT_COLOR = cv::Scalar(0, 0, 255);
	const cv::Scalar TXT_COLOR = cv::Scalar(255, 255, 255);

	const char* INPUT = "images";
	const char* OUTPUT = "outputs";
	const char* PROTO = "proto";

	const char* CLASS_NAMES[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus",
		"train", "truck", "boat", "traffic light", "fire hydrant",
		"stop sign", "parking meter", "bench", "bird", "cat",
		"dog", "horse", "sheep", "cow", "elephant",
		"bear", "zebra", "giraffe", "backpack", "umbrella",
		"handbag", "tie", "suitcase", "frisbee", "skis",
		"snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
		"skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
		"cup", "fork", "knife", "spoon", "bowl",
		"banana", "apple", "sandwich", "orange", "broccoli",
		"carrot", "hot dog", "pizza", "donut", "cake",
		"chair", "couch", "potted plant", "bed", "dining table",
		"toilet", "tv", "laptop", "mouse", "remote",
		"keyboard", "cell phone", "microwave", "oven",
		"toaster", "sink", "refrigerator", "book", "clock", "vase",
		"scissors", "teddy bear", "hair drier", "toothbrush" };

	const unsigned int COLORS[80][3] = {
		{ 0, 114, 189 }, { 217, 83, 25 }, { 237, 177, 32 },
		{ 126, 47, 142 }, { 119, 172, 48 }, { 77, 190, 238 },
		{ 162, 20, 47 }, { 76, 76, 76 }, { 153, 153, 153 },
		{ 255, 0, 0 }, { 255, 128, 0 }, { 191, 191, 0 },
		{ 0, 255, 0 }, { 0, 0, 255 }, { 170, 0, 255 },
		{ 85, 85, 0 }, { 85, 170, 0 }, { 85, 255, 0 },
		{ 170, 85, 0 }, { 170, 170, 0 }, { 170, 255, 0 },
		{ 255, 85, 0 }, { 255, 170, 0 }, { 255, 255, 0 },
		{ 0, 85, 128 }, { 0, 170, 128 }, { 0, 255, 128 },
		{ 85, 0, 128 }, { 85, 85, 128 }, { 85, 170, 128 },
		{ 85, 255, 128 }, { 170, 0, 128 }, { 170, 85, 128 },
		{ 170, 170, 128 }, { 170, 255, 128 }, { 255, 0, 128 },
		{ 255, 85, 128 }, { 255, 170, 128 }, { 255, 255, 128 },
		{ 0, 85, 255 }, { 0, 170, 255 }, { 0, 255, 255 },
		{ 85, 0, 255 }, { 85, 85, 255 }, { 85, 170, 255 },
		{ 85, 255, 255 }, { 170, 0, 255 }, { 170, 85, 255 },
		{ 170, 170, 255 }, { 170, 255, 255 }, { 255, 0, 255 },
		{ 255, 85, 255 }, { 255, 170, 255 }, { 85, 0, 0 },
		{ 128, 0, 0 }, { 170, 0, 0 }, { 212, 0, 0 },
		{ 255, 0, 0 }, { 0, 43, 0 }, { 0, 85, 0 },
		{ 0, 128, 0 }, { 0, 170, 0 }, { 0, 212, 0 },
		{ 0, 255, 0 }, { 0, 0, 43 }, { 0, 0, 85 },
		{ 0, 0, 128 }, { 0, 0, 170 }, { 0, 0, 212 },
		{ 0, 0, 255 }, { 0, 0, 0 }, { 36, 36, 36 },
		{ 73, 73, 73 }, { 109, 109, 109 }, { 146, 146, 146 },
		{ 182, 182, 182 }, { 219, 219, 219 }, { 0, 114, 189 },
		{ 80, 183, 189 }, { 128, 128, 0 }
	};

	const unsigned int MASK_COLORS[20][3] = {
		{ 255, 56, 56 }, { 255, 157, 151 }, { 255, 112, 31 },
		{ 255, 178, 29 }, { 207, 210, 49 }, { 72, 249, 10 },
		{ 146, 204, 23 }, { 61, 219, 134 }, { 26, 147, 52 },
		{ 0, 212, 187 }, { 44, 153, 168 }, { 0, 194, 255 },
		{ 52, 69, 147 }, { 100, 115, 255 }, { 0, 24, 236 },
		{ 132, 56, 255 }, { 82, 0, 133 }, { 203, 56, 255 },
		{ 255, 149, 200 }, { 255, 55, 199 }
	};

	struct Object
	{
		cv::Rect_<float> rect;
		int label = 0;
		float prob = 0.0;
		cv::Mat boxMask;
	};

}
#endif //YOLOV8_TENSORRT_CSRC_SEGMENT_INCLUDE_CONFIG_H
