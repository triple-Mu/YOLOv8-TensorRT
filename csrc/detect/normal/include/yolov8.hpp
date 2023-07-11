//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs,
                                     float                score_thres = 0.25f,
                                     float                iou_thres   = 0.65f,
                                     int                  topk        = 100,
                                     int                  num_labels  = 80);
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
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
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
        }
        else {
            dims         = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
#endif  // DETECT_NORMAL_YOLOV8_HPP
