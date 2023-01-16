//
// Created by ubuntu on 1/8/23.
//
#include "include/yolov8-seg.hpp"
int main(int argc, char** argv)
{
	cudaSetDevice(DEVICE);

	const std::string engine_file_path{ argv[1] };
	const std::string path{ argv[2] };
	std::vector<cv::String> imagePathList;
	bool isVideo{ false };
	if (IsFile(path))
	{
		std::string suffix = path.substr(path.find_last_of('.') + 1);
		if (suffix == "jpg")
		{
			imagePathList.push_back(path);
		}
		else if (suffix == "mp4")
		{
			isVideo = true;
		}
	}
	else if (IsFolder(path))
	{
		cv::glob(path + "/*.jpg", imagePathList);
	}

	auto* yolov8 = new YOLOv8_seg(engine_file_path);
	yolov8->make_pipe(true);
	cv::Mat res;
	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	if (isVideo)
	{
		cv::VideoCapture cap(path);
		cv::Mat image;
		if (!cap.isOpened())
		{
			printf("can not open ...\n");
			return -1;
		}
		double fp_ = cap.get(cv::CAP_PROP_FPS);
		int fps = round(1000.0 / fp_);
		while (cap.read(image))
		{
			auto start = std::chrono::system_clock::now();
			yolov8->copy_from_Mat(image);
			yolov8->infer();
			std::vector<Object> objs;
			yolov8->postprocess(objs);
			draw_objects(image, res, objs);
			auto end = std::chrono::system_clock::now();
			auto tc = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
			cv::imshow("result", res);
			printf("cost %2.4f ms\n", tc);
			if (cv::waitKey(fps) == 'q')
			{
				break;
			}
		}
	}
	else
	{
		for (auto path : imagePathList)
		{
			cv::Mat image = cv::imread(path);
			yolov8->copy_from_Mat(image);
			auto start = std::chrono::system_clock::now();
			yolov8->infer();
			auto end = std::chrono::system_clock::now();
			auto tc = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;

			printf("infer %-20s\tcost %2.4f ms\n", path.c_str(), tc);

			std::vector<Object> objs;
			yolov8->postprocess(objs);
			draw_objects(image, res, objs);
			cv::imshow("result", res);
			cv::waitKey(0);
		}
	}
	cv::destroyAllWindows();
	delete yolov8;
	return 0;
}
