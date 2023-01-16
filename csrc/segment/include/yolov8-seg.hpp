//
// Created by ubuntu on 1/8/23.
//
#include "config.h"
#include "utils.h"
#include <fstream>
#include "NvInferPlugin.h"

using namespace seg;

class YOLOv8_seg
{
public:
	explicit YOLOv8_seg(const std::string& engine_file_path);
	~YOLOv8_seg();

	void make_pipe(bool warmup = true);
	void copy_from_Mat(const cv::Mat& image);
	void infer();
	void postprocess(std::vector<Object>& objs);

	size_t in_size = 1 * 3 * INPUT_W * INPUT_H;
	float w = INPUT_W;
	float h = INPUT_H;
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

YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path)
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

YOLOv8_seg::~YOLOv8_seg()
{
	this->context->destroy();
	this->engine->destroy();
	this->runtime->destroy();
	cudaStreamDestroy(this->stream);
	for (auto& ptr : this->buffs)
	{
		CHECK(cudaFree(ptr));
	}

	for (auto& ptr : this->outputs)
	{
		CHECK(cudaFreeHost(ptr));
	}

}
void YOLOv8_seg::make_pipe(bool warmup)
{
	const nvinfer1::Dims input_dims = this->engine->getBindingDimensions(
		this->engine->getBindingIndex(INPUT)
	);
	this->in_size = get_size_by_dims(input_dims);
	CHECK(cudaMalloc(&this->buffs[0], this->in_size * sizeof(float)));

	this->context->setBindingDimensions(0, input_dims);

	const int32_t output_idx = this->engine->getBindingIndex(OUTPUT);
	const nvinfer1::Dims output_dims = this->context->getBindingDimensions(output_idx);
	this->out_sizes[output_idx - NUM_INPUT].first = get_size_by_dims(output_dims);
	this->out_sizes[output_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(output_idx));

	const int32_t proto_idx = this->engine->getBindingIndex(PROTO);
	const nvinfer1::Dims proto_dims = this->context->getBindingDimensions(proto_idx);

	this->out_sizes[proto_idx - NUM_INPUT].first = get_size_by_dims(proto_dims);
	this->out_sizes[proto_idx - NUM_INPUT].second = DataTypeToSize(
		this->engine->getBindingDataType(proto_idx));

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

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image)
{
	float height = (float)image.rows;
	float width = (float)image.cols;

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

void YOLOv8_seg::infer()
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

void YOLOv8_seg::postprocess(std::vector<Object>& objs)
{
	objs.clear();
	auto* output = static_cast<float*>(this->outputs[0]); // x0 y0 x1 y1 s l *32
	cv::Mat protos = cv::Mat(NUM_SEG_C, SEG_W * SEG_H, CV_32F,
		static_cast<float*>(this->outputs[1]));

	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<cv::Rect> bboxes;
	std::vector<cv::Mat> mask_confs;

	for (int i = 0; i < NUM_PROPOSAL; i++)
	{
		float* ptr = output + i * NUM_COLS;
		float score = *(ptr + 4);
		if (score > CONF_THRES)
		{
			float x0 = *ptr++ - this->dw;
			float y0 = *ptr++ - this->dh;
			float x1 = *ptr++ - this->dw;
			float y1 = *ptr++ - this->dh;

			x0 = clamp(x0 * this->ratio, 0.f, this->w);
			y0 = clamp(y0 * this->ratio, 0.f, this->h);
			x1 = clamp(x1 * this->ratio, 0.f, this->w);
			y1 = clamp(y1 * this->ratio, 0.f, this->h);

			int label = *(++ptr);
			cv::Mat mask_conf = cv::Mat(1, NUM_SEG_C, CV_32F, ++ptr);
			mask_confs.push_back(mask_conf);
			labels.push_back(label);
			scores.push_back(score);

#if defined(BATCHED_NMS)
			bboxes.push_back(cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0));
#else
			bboxes.push_back(cv::Rect_<float>(x0 + label * DIS,
				y0 + label * DIS,
				x1 - x0,
				y1 - y0));
#endif
		}
	}
	std::vector<int> indices;
#if defined(BATCHED_NMS)
	cv::dnn::NMSBoxesBatched(bboxes, scores, labels, CONF_THRES, IOU_THRES, indices);
#else
	cv::dnn::NMSBoxes(bboxes, scores, CONF_THRES, IOU_THRES, indices);
#endif

	cv::Mat masks;

	for (auto& i : indices)
	{
#if defined(BATCHED_NMS)
		cv::Rect tmp = bboxes[i];
#else
		cv::Rect tmp = { (int)(bboxes[i].x - labels[i] * DIS),
						 (int)(bboxes[i].y - labels[i] * DIS),
						 bboxes[i].width,
						 bboxes[i].height };
#endif

		Object obj;
		obj.label = labels[i];
		obj.rect = tmp;
		obj.prob = scores[i];
		masks.push_back(mask_confs[i]);
		objs.push_back(obj);
	}

	cv::Mat matmulRes = (masks * protos).t();
	cv::Mat maskMat = matmulRes.reshape(indices.size(), { SEG_W, SEG_H });

	std::vector<cv::Mat> maskChannels;
	cv::split(maskMat, maskChannels);
	int scale_dw = this->dw / INPUT_W * SEG_W;
	int scale_dh = this->dh / INPUT_H * SEG_H;

	cv::Rect roi(
		scale_dw,
		scale_dh,
		SEG_W - 2 * scale_dw,
		SEG_H - 2 * scale_dh);

	for (int i = 0; i < indices.size(); i++)
	{
		cv::Mat dest, mask;
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);
		dest = dest(roi);
		cv::resize(dest, mask, cv::Size((int)this->w, (int)this->h), cv::INTER_LINEAR);
		objs[i].boxMask = mask(objs[i].rect) > MASK_THRES;
	}

}

static void draw_objects(const cv::Mat& image, cv::Mat& res, const std::vector<Object>& objs)
{
	res = image.clone();
	cv::Mat mask = image.clone();
	for (auto& obj : objs)
	{
		int idx = obj.label;
		cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
		cv::Scalar mask_color = cv::Scalar(
			MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
		cv::rectangle(res, obj.rect, color, 2);

		char text[256];
		sprintf(text, "%s %.1f%%", CLASS_NAMES[idx], obj.prob * 100);
		mask(obj.rect).setTo(mask_color, obj.boxMask);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

		int x = (int)obj.rect.x;
		int y = (int)obj.rect.y + 1;

		if (y > res.rows)
			y = res.rows;

		cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), RECT_COLOR, -1);

		cv::putText(res, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.4, TXT_COLOR, 1);
	}
	cv::addWeighted(res, 0.5, mask, 0.8, 1, res);

}
