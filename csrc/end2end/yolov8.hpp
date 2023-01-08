//
// Created by ubuntu on 1/8/23.
//
#include "detect_config.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <chrono>
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

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

const char* CLASS_NAMES[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
							  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
							  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
							  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
							  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
							  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
							  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
							  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
							  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
							  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
							  "toothbrush" };

const unsigned int COLORS[80][3] = {
	{ 0, 114, 189 },
	{ 217, 83, 25 },
	{ 237, 177, 32 },
	{ 126, 47, 142 },
	{ 119, 172, 48 },
	{ 77, 190, 238 },
	{ 162, 20, 47 },
	{ 76, 76, 76 },
	{ 153, 153, 153 },
	{ 255, 0, 0 },
	{ 255, 128, 0 },
	{ 191, 191, 0 },
	{ 0, 255, 0 },
	{ 0, 0, 255 },
	{ 170, 0, 255 },
	{ 85, 85, 0 },
	{ 85, 170, 0 },
	{ 85, 255, 0 },
	{ 170, 85, 0 },
	{ 170, 170, 0 },
	{ 170, 255, 0 },
	{ 255, 85, 0 },
	{ 255, 170, 0 },
	{ 255, 255, 0 },
	{ 0, 85, 128 },
	{ 0, 170, 128 },
	{ 0, 255, 128 },
	{ 85, 0, 128 },
	{ 85, 85, 128 },
	{ 85, 170, 128 },
	{ 85, 255, 128 },
	{ 170, 0, 128 },
	{ 170, 85, 128 },
	{ 170, 170, 128 },
	{ 170, 255, 128 },
	{ 255, 0, 128 },
	{ 255, 85, 128 },
	{ 255, 170, 128 },
	{ 255, 255, 128 },
	{ 0, 85, 255 },
	{ 0, 170, 255 },
	{ 0, 255, 255 },
	{ 85, 0, 255 },
	{ 85, 85, 255 },
	{ 85, 170, 255 },
	{ 85, 255, 255 },
	{ 170, 0, 255 },
	{ 170, 85, 255 },
	{ 170, 170, 255 },
	{ 170, 255, 255 },
	{ 255, 0, 255 },
	{ 255, 85, 255 },
	{ 255, 170, 255 },
	{ 85, 0, 0 },
	{ 128, 0, 0 },
	{ 170, 0, 0 },
	{ 212, 0, 0 },
	{ 255, 0, 0 },
	{ 0, 43, 0 },
	{ 0, 85, 0 },
	{ 0, 128, 0 },
	{ 0, 170, 0 },
	{ 0, 212, 0 },
	{ 0, 255, 0 },
	{ 0, 0, 43 },
	{ 0, 0, 85 },
	{ 0, 0, 128 },
	{ 0, 0, 170 },
	{ 0, 0, 212 },
	{ 0, 0, 255 },
	{ 0, 0, 0 },
	{ 36, 36, 36 },
	{ 73, 73, 73 },
	{ 109, 109, 109 },
	{ 146, 146, 146 },
	{ 182, 182, 182 },
	{ 219, 219, 219 },
	{ 0, 114, 189 },
	{ 80, 183, 189 },
	{ 128, 128, 0 }
};



struct Object
{
	cv::Rect_<float> rect;
	int label = 0;
	float prob = 0.0;
};

class YOLOv8
{
public:
	explicit YOLOv8(const std::string& engine_file_path);
	~YOLOv8();

	void make_pipe(bool warmup = true);
	void copy_from_Mat(const cv::Mat& image);
	void infer();
	void postprocess(std::vector<Object>& objs);

	size_t in_size = 1 * 3 * INPUT_W * INPUT_H;
	float w;
	float h;
	float ratio = 1.0f;
	float dw = 0.f;
	float dh = 0.f;
	std::array<std::pair<int, int>, NUM_OUTPUT> out_sizes{};
	std::array<void*, NUM_OUTPUT> outputs{};
private:
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
	cudaStream_t stream = nullptr;
	std::array<void*, NUM_BINDINGS> buffs{};
	Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

};

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
	std::ifstream file(engine_file_path, std::ios::binary);
	assert(file.good());
	file.seekg(0, std::ios::end);
	auto size = file.tellg();
	std::ostringstream fmt;

	file.seekg(0, std::ios::beg);
	char* trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();
	initLibNvInferPlugins(&this->gLogger, "");
	this->runtime = nvinfer1::createInferRuntime(this->gLogger);
	assert(this->runtime != nullptr);

	this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);

	this->context = this->engine->createExecutionContext();

	assert(this->context != nullptr);
	cudaStreamCreate(&this->stream);

}

YOLOv8::~YOLOv8()
{
	this->engine->destroy();
	this->context->destroy();
	this->runtime->destroy();
	cudaStreamDestroy(this->stream);
	for (auto& ptr : this->buffs)
	{
		CHECK(cudaFree(ptr));
	}

	for (auto& ptr : this->outputs)
	{
		CHECK(cudaFree(ptr));
	}

}
void YOLOv8::make_pipe(bool warmup)
{
	const nvinfer1::Dims input_dims = this->engine->getBindingDimensions(
		this->engine->getBindingIndex(INPUT)
	);
	this->in_size = get_size_by_dims(input_dims);
	CHECK(cudaMalloc(&this->buffs[0], this->in_size * sizeof(float)));

	this->context->setBindingDimensions(0, input_dims);
	const int32_t num_dets_idx = this->engine->getBindingIndex(NUM_DETS);
	const nvinfer1::Dims num_dets_dims = this->context->getBindingDimensions(num_dets_idx);
	this->out_sizes[num_dets_idx - NUM_INPUT].first = get_size_by_dims(num_dets_dims);
	this->out_sizes[num_dets_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(num_dets_idx));

	const int32_t bboxes_idx = this->engine->getBindingIndex(BBOXES);
	const nvinfer1::Dims bboxes_dims = this->context->getBindingDimensions(bboxes_idx);

	this->out_sizes[bboxes_idx - NUM_INPUT].first = get_size_by_dims(bboxes_dims);
	this->out_sizes[bboxes_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(bboxes_idx));

	const int32_t scores_idx = this->engine->getBindingIndex(SCORES);
	const nvinfer1::Dims scores_dims = this->context->getBindingDimensions(scores_idx);
	this->out_sizes[scores_idx - NUM_INPUT].first = get_size_by_dims(scores_dims);
	this->out_sizes[scores_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(scores_idx));

	const int32_t labels_idx = this->engine->getBindingIndex(LABELS);
	const nvinfer1::Dims labels_dims = this->context->getBindingDimensions(labels_idx);
	this->out_sizes[labels_idx - NUM_INPUT].first = get_size_by_dims(labels_dims);
	this->out_sizes[labels_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(labels_idx));

	for (int i = 0; i < NUM_OUTPUT; i++)
	{
		const int osize = this->out_sizes[i].first * out_sizes[i].second;
		CHECK(cudaHostAlloc(&this->outputs[i], osize, 0));
		CHECK(cudaMalloc(&this->buffs[NUM_INPUT + i], osize));
	}
	if (warmup)
	{
		for (int i = 0; i < 10; i++)
		{
			size_t isize = this->in_size * sizeof(float);
			auto* tmp = new float[isize];

			CHECK(cudaMemcpyAsync(this->buffs[0],
				tmp,
				isize,
				cudaMemcpyHostToDevice,
				this->stream));
			this->infer();
		}
		printf("model warmup 10 times\n");

	}
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
	float height = image.rows;
	float width = image.cols;

	float r = std::min(INPUT_H / height, INPUT_W / width);

	int padw = (int)std::round(width * r);
	int padh = (int)std::round(height * r);

	cv::Mat tmp;
	if ((int)width != padw || (int)height != padh)
	{
		cv::resize(image, tmp, cv::Size(padw, padh));
	}
	else
	{
		tmp = image.clone();
	}

	float _dw = INPUT_W - padw;
	float _dh = INPUT_H - padh;

	_dw /= 2.0f;
	_dh /= 2.0f;
	int top = int(std::round(_dh - 0.1f));
	int bottom = int(std::round(_dh + 0.1f));
	int left = int(std::round(_dw - 0.1f));
	int right = int(std::round(_dw + 0.1f));
	cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, PAD_COLOR);
	cv::dnn::blobFromImage(tmp,
		tmp,
		1 / 255.f,
		cv::Size(),
		cv::Scalar(0, 0, 0),
		true,
		false,
		CV_32F);
	CHECK(cudaMemcpyAsync(this->buffs[0],
		tmp.ptr<float>(),
		this->in_size * sizeof(float),
		cudaMemcpyHostToDevice,
		this->stream));

	this->ratio = 1 / r;
	this->dw = _dw;
	this->dh = _dh;
	this->w = width;
	this->h = height;
}

void YOLOv8::infer()
{
	this->context->enqueueV2(buffs.data(), this->stream, nullptr);
	for (int i = 0; i < NUM_OUTPUT; i++)
	{
		const int osize = this->out_sizes[i].first * out_sizes[i].second;
		CHECK(cudaMemcpyAsync(this->outputs[i],
			this->buffs[NUM_INPUT + i],
			osize,
			cudaMemcpyDeviceToHost,
			this->stream));
	}
	cudaStreamSynchronize(this->stream);

}

void YOLOv8::postprocess(std::vector<Object>& objs)
{
	int* num_dets = static_cast<int*>(this->outputs[0]);
	auto* boxes = static_cast<float*>(this->outputs[1]);
	auto* scores = static_cast<float*>(this->outputs[2]);
	int* labels = static_cast<int*>(this->outputs[3]);
	for (int i = 0; i < num_dets[0]; i++)
	{
		Object obj;
		float x0 = (boxes[i * 4]) - this->dw;
		float y0 = (boxes[i * 4 + 1]) - this->dh;
		float x1 = (boxes[i * 4 + 2]) - this->dw;
		float y1 = (boxes[i * 4 + 3]) - this->dh;

		x0 = clamp(x0 * this->ratio, 0.f, this->w);
		y0 = clamp(y0 * this->ratio, 0.f, this->h);
		x1 = clamp(x1 * this->ratio, 0.f, this->w);
		y1 = clamp(y1 * this->ratio, 0.f, this->h);
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = scores[i];
		obj.label = labels[i];

		objs.push_back(obj);

	}
}

static void draw_objects(const cv::Mat& image, const std::vector<Object>& objs)
{
	cv::Mat img = image.clone();
	for (auto& obj : objs)
	{
		cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
		cv::rectangle(img, obj.rect, color, 2);

		char text[256];
		sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		int x = (int)obj.rect.x;
		int y = (int)obj.rect.y + 1;

		if (y > img.rows)
			y = img.rows;

		cv::rectangle(img, cv::Rect(x, y, label_size.width, label_size.height + baseLine), RECT_COLOR, -1);

		cv::putText(img, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.4, TXT_COLOR, 1);
	}

	cv::imshow("results", img);
}
