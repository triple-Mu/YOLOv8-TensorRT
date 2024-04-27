//
// Created by ubuntu on 4/27/24.
//
#ifndef CLS_NORMAL_YOLOv8_cls_HPP
#define CLS_NORMAL_YOLOv8_cls_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"

using namespace cls;

class YOLOv8_cls {
public:
    explicit YOLOv8_cls(const std::string& engine_file_path);

    ~YOLOv8_cls();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs);

    static void draw_objects(const cv::Mat&                  image,
                             cv::Mat&                        res,
                             const std::vector<Object>&      objs,
                             const std::vector<std::string>& CLASS_NAMES);

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8_cls::YOLOv8_cls(const std::string& engine_file_path)
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

YOLOv8_cls::~YOLOv8_cls()
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

void YOLOv8_cls::make_pipe(bool warmup)
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

void YOLOv8_cls::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat nchw;
    auto&   in_binding = this->input_bindings[0];
    auto    width      = in_binding.dims.d[3];
    auto    height     = in_binding.dims.d[2];

    cv::dnn::blobFromImage(image, nchw, 1 / 255.f, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false, CV_32F);

    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_cls::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    cv::dnn::blobFromImage(image, nchw, 1 / 255.f, size, cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_cls::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_cls::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    auto num_cls = this->output_bindings[0].dims.d[1];

    float* max_ptr =
        std::max_element(static_cast<float*>(this->host_ptrs[0]), static_cast<float*>(this->host_ptrs[0]) + num_cls);
    Object obj;
    obj.label = std::distance(static_cast<float*>(this->host_ptrs[0]), max_ptr);
    obj.prob  = *max_ptr;
    objs.push_back(obj);
}

void YOLOv8_cls::draw_objects(const cv::Mat&                  image,
                              cv::Mat&                        res,
                              const std::vector<Object>&      objs,
                              const std::vector<std::string>& CLASS_NAMES)
{
    res = image.clone();
    char   text[256];
    Object obj = objs[0];
    sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
    int      x          = 10;
    int      y          = 10;

    if (y > res.rows)
        y = res.rows;

    cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
    cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
}

#endif  // CLS_NORMAL_YOLOv8_cls_HPP
