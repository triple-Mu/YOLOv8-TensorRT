//
// Created by ubuntu on 2/8/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8-seg.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const fs::path    path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 3);

    auto yolov8_seg = new YOLOv8_seg(engine_file_path);
    yolov8_seg->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size         = cv::Size{640, 640};
    int      topk         = 100;
    int      seg_h        = 160;
    int      seg_w        = 160;
    int      seg_channels = 32;
    float    score_thres  = 0.25f;
    float    iou_thres    = 0.65f;

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8_seg->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_seg->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_seg->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8_seg->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_seg->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_seg->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_seg->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);
            yolov8_seg->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8_seg;
    return 0;
}
