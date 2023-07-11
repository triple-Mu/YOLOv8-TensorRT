//
// Created by ubuntu on 4/7/23.
//
#include "chrono"
#include "opencv2/opencv.hpp"
#include "yolov8-pose.hpp"

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};

int main(int argc, char** argv)
{
    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    assert(argc == 3);

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size        = cv::Size{640, 640};
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
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& path : imagePathList) {
            objs.clear();
            image = cv::imread(path);
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8_pose;
    return 0;
}
