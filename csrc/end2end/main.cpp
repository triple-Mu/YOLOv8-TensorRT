//
// Created by ubuntu on 1/8/23.
//
#include "yolov8.hpp"
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

	auto* yolov8 = new YOLOv8(engine_file_path);
	yolov8->make_pipe(true);

	if (isVideo)
	{
		cv::VideoCapture cap(path);
		cv::Mat image;

		while (cap.isOpened())
		{
			cap >> image;
			yolov8->copy_from_Mat(image);
			yolov8->infer();
			std::vector<Object> objs;
			yolov8->postprocess(objs);
			draw_objects(image, objs);
			if (cv::waitKey(1) == 'q')
			{
				break;
			}

		}
		cv::destroyAllWindows();
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
			draw_objects(image, objs);
			cv::waitKey(0);
		}
	}
	delete yolov8;
	return 0;
}
