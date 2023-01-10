//
// Created by ubuntu on 1/10/23.
//

#ifndef YOLOV8_CSRC_END2END_INCLUDE_CONFIG_H
#define YOLOV8_CSRC_END2END_INCLUDE_CONFIG_H
#include "opencv2/opencv.hpp"
namespace det
{
	const int DEVICE = 0;

	static const int INPUT_W = 640;
	static const int INPUT_H = 640;
	static const int NUM_INPUT = 1;
	static const int NUM_OUTPUT = 4;

	static const int NUM_BINDINGS = NUM_INPUT + NUM_OUTPUT;
	const cv::Scalar PAD_COLOR = { 114, 114, 114 };
	const cv::Scalar RECT_COLOR = cv::Scalar(0, 0, 255);
	const cv::Scalar TXT_COLOR = cv::Scalar(255, 255, 255);

	const char* INPUT = "images";
	const char* NUM_DETS = "num_dets";
	const char* BBOXES = "bboxes";
	const char* SCORES = "scores";
	const char* LABELS = "labels";

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

	struct Object
	{
		cv::Rect_<float> rect;
		int label = 0;
		float prob = 0.0;
	};

}
#endif //YOLOV8_CSRC_END2END_INCLUDE_CONFIG_H
