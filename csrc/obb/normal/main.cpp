//
// Created by ubuntu on 4/7/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8-obb.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {"plane",
                                              "ship",
                                              "storage tank",
                                              "baseball diamond",
                                              "tennis court",
                                              "basketball court",
                                              "ground track field",
                                              "harbor",
                                              "bridge",
                                              "large vehicle",
                                              "small vehicle",
                                              "helicopter",
                                              "roundabout",
                                              "soccer ball field",
                                              "swimming pool"};

const std::vector<std::vector<unsigned int>> COLORS = {{0, 114, 189},
                                                       {217, 83, 25},
                                                       {237, 177, 32},
                                                       {126, 47, 142},
                                                       {119, 172, 48},
                                                       {77, 190, 238},
                                                       {162, 20, 47},
                                                       {76, 76, 76},
                                                       {153, 153, 153},
                                                       {255, 0, 0},
                                                       {255, 128, 0},
                                                       {191, 191, 0},
                                                       {0, 255, 0},
                                                       {0, 0, 255},
                                                       {170, 0, 255}};

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

    auto yolov8_obb = new YOLOv8_obb(engine_file_path);
    yolov8_obb->make_pipe(true);

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
    cv::Size size        = cv::Size{1024, 1024};
    int      num_labels  = 15;
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.65f;

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
            yolov8_obb->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_obb->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_obb->postprocess(objs, score_thres, iou_thres, topk, num_labels);
            yolov8_obb->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
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
            yolov8_obb->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_obb->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_obb->postprocess(objs, score_thres, iou_thres, topk, num_labels);
            yolov8_obb->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8_obb;
    return 0;
}
