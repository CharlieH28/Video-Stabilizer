#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


namespace cv {
    class Assignment {
    public:
        void core();
        void completion();
        void challenge();
    };

    class SIFTcore {
    public:
        void core1(Mat img_1, Mat img_2);
        void match();     
        void core2(Mat img1, Mat img2);
        void core3(Mat img1, Mat img2);
    };

    class VideoStabilizer {
    public:
        Mat extraction(Mat img_1, Mat img_2);
        std::vector<Mat> loadImages(int numFrames,const String prefix);
        void exportImages(const std::vector<Mat>& images, const String prefix, const String outputFolder);
        void generate1DGaussian(double mean, double stddev, int size, std::vector<double>& gaussian);
        void videoStabilization(std::vector<Mat>& frames);
        std::vector<Mat> stabilizedExtraction(std::vector<Mat>& frames);
        void minCrop(std::vector<Mat>& frames);
    };

    class gray {
    public:
        void Grayswork();
    };
}