#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "Source.h"
#include <opencv2/features2d.hpp>
#include <random>


/*
    Assignment Section -> all the required functions are implemented here.

*/

int main() {
    
    std::cout << "Assignment 4 begins...\n";
    // load images

    //Perform assignment sections
    cv::Assignment A;
    A.core();
    A.completion();
    A.challenge();
    
    std::cout << "...Assignment 4 complete\n";
    cv::waitKey(0);
    return 0; }

void cv::Assignment::core() {
    
   Mat img_039 = imread("Input_A4/Frame039.jpg",1);
   Mat img_041 = imread("Input_A4/Frame041.jpg",1);

   SIFTcore S;

   // Reveal Part 1
   S.core1(img_039, img_041);
   
   // Reveal Part 2   
   S.core2(img_039, img_041);

   //Reveal Part 3;
   S.core3(img_039, img_041);
}


/*
    CORE Class Section -> all required CORE classes implemented here.

*/
void cv::SIFTcore::core1(Mat img_1,Mat img_2) {

    // Initialize SIFT
    Ptr<cv::SIFT> sift = sift->create();
    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;

    Mat descriptors_1, descriptors_2;

    sift->detectAndCompute(img_1, noArray(),keypoints_1, descriptors_1);
    sift->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    // Find matches
    BFMatcher matcher(NORM_L2, true);
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    Mat result;
    vconcat(img_1, img_2, result);

    // Draw lines between matches 
    for (int i = 0; i < matches.size(); i++)
    {
        Point2f p1 = keypoints_1[matches[i].queryIdx].pt;
        Point2f p2 = keypoints_2[matches[i].trainIdx].pt + Point2f(0, img_1.rows);
        line(result, p1, p2, Scalar(0, 255, 0), 1);
    }

    imshow("Matches core 1", result);
   
}

void cv::SIFTcore::core2(Mat img_1, Mat img_2) {

    // 1. Generate feature pairs using SIFT (core 1)
    Ptr<cv::SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keypoints_1;
    std::vector<KeyPoint> keypoints_2;
    Mat descriptors_1, descriptors_2;

    sift->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    sift->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    BFMatcher matcher(NORM_L2, true);
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);


    std::vector<DMatch> edges = matches;
    std::vector<DMatch> bestInliers;
    Mat bestHomography;
    srand(time(NULL));
    int epsilon = 10;

    // 2. Iterate a fixed number of times, e.g. 100:
    for (int i = 0; i < 100; i++) {
        std::cout << i << std::endl;
        std::cout << "--> inter-computing analysis 1: -- Select 4 feature pairs at random\n";
        // create the inlier list
        std::vector<DMatch> inlierEdges;
        std::vector<DMatch> outlierEdges;

        // a) Select 4 feature pairs at random. Make sure they are different random pairs
        int rand1 = rand() % edges.size();
        int rand2 = rand() % edges.size();
        int rand3 = rand() % edges.size();
        int rand4 = rand() % edges.size();


        while (rand2 == rand1 || rand2 == rand3 || rand2 == rand4)
        {
            
            rand2 = rand() % edges.size();
        }
        while (rand3 == rand1 || rand3 == rand2 || rand3 == rand4)
        {
            
            rand3 = rand() % edges.size();
        }
        while (rand4 == rand1 || rand4 == rand2 || rand4 == rand3)
        {
            
            rand4 = rand() % edges.size();
        }

        std::vector<Point2f> src;
        src.push_back(keypoints_1[edges[rand1].queryIdx].pt);
        src.push_back(keypoints_1[edges[rand2].queryIdx].pt);
        src.push_back(keypoints_1[edges[rand3].queryIdx].pt);
        src.push_back(keypoints_1[edges[rand4].queryIdx].pt);

        std::vector<Point2f> dst;
        dst.push_back(keypoints_2[edges[rand1].trainIdx].pt);
        dst.push_back(keypoints_2[edges[rand2].trainIdx].pt);
        dst.push_back(keypoints_2[edges[rand3].trainIdx].pt);
        dst.push_back(keypoints_2[edges[rand4].trainIdx].pt);

        std::cout << "--> inter-computing analysis 2: -- Compute the homography transform H for the pairs exactly.\n";

        // b) Compute the homography transform H for the pairs exactly.
        Mat h = findHomography(src, dst, 0);
        cv::Mat_<float> hMat = cv::Mat::eye(3, 3, CV_32FC1);

        // c) Compute inliers amongst all of the pairs, where the mapping error of the transformed 
        // point q with the target position p is less than some epsilon.
        std::cout << "--> inter-computing analysis 3: -- computing inliers and outliers\n";
        //compute inliers and outliers
        for (DMatch e : edges) {
            if (!h.empty()) {
                hMat = h;

                // get the edge point in the matches
                Vec3f point(keypoints_1[e.queryIdx].pt.x, keypoints_1[e.queryIdx].pt.y, 1);

                // calculate the error from the homography and the edge
                Mat q = hMat * Mat(point);
                Vec3f qVec = q;
                Point2f pointQ = Point(qVec[0], qVec[1]);
                float error = norm((Mat)keypoints_2[e.trainIdx].pt, (Mat)pointQ);

                

                // check if error is smaller than the epsilon
                if (error < epsilon) {
                    inlierEdges.push_back(e);
                }
            }
        }
        std::cout << "--> inter-computing analysis 4: -- Homography analysis \n";
        std::cout << "inlier edge side " << inlierEdges.size() << "  best inlier size " << bestInliers.size() << std::endl;
        
        // d) If the number of inlier pairs is greater than the previous iteration's, save the current 
        // homography transform H and the number of inlier pairs.
        if (inlierEdges.size() > bestInliers.size()) {
            bestInliers = inlierEdges;
            bestHomography = h.clone();
        }
        std::cout << "--> inter-computing analysis end\n";
        std::cout << "--> computing iteration number "<<i<<" \n";
    }
    std::cout << "    ...finished computing \n";
    
    
    std::vector<DMatch> finalInliers;
    std::vector<DMatch> finalOutliers;
    

    for (DMatch e : edges) {
        Mat_<float> hMat = Mat::eye(3, 3, CV_32FC1);
        hMat = bestHomography;

        //find the edge point
        Vec3f point(keypoints_1[e.queryIdx].pt.x, keypoints_1[e.queryIdx].pt.y, 1);

        // calculate error
        Mat q = hMat * Mat(point);
        Vec3f qVec = q;
        Point2f pointQ = Point(qVec[0], qVec[1]);
        float error = norm((Mat)keypoints_2[e.trainIdx].pt, (Mat)pointQ);

        // check whether the error is smaller than the epsilon else add it to the outliers
        if (error < epsilon) {
            finalInliers.push_back(e);
        }
        else {
            finalOutliers.push_back(e);
        }
    }

    std::cout << "inlier size " << finalInliers.size() << "  outlier size " << finalOutliers.size() << std::endl;
    Mat result;
    vconcat(img_1, img_2, result);

    //draw inliners in green
    for (DMatch in : finalInliers) {
        line(result, Point(keypoints_1[in.queryIdx].pt), Point(keypoints_2[in.trainIdx].pt) + Point(0, img_1.rows), Scalar(0, 255, 0), 1);
    }
    //draw outliers in red
    for (DMatch out : finalOutliers) {
        line(result, Point(keypoints_1[out.queryIdx].pt), Point(keypoints_2[out.trainIdx].pt) + Point(0, img_1.rows), Scalar(0, 0, 255), 1);
    }

    imshow("Matches core 2", result);

}
void cv::SIFTcore::core3(Mat img1, Mat img2) {

    // Convert the images to grayscale
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // Detect keypoints and compute descriptors using ORB
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Ptr<ORB> sift = ORB::create();
    sift->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

    // Match the descriptors using Brute-Force Matcher
    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find the homography matrix using RANSAC
    std::vector<Point2f> points1, points2; 
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    Mat H = findHomography(points1, points2, RANSAC);

    //add 100px padding to each image
    Mat padded1, padded2;
    copyMakeBorder(img1, padded1, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
    copyMakeBorder(img2, padded2, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));


    Mat warped1;
    warpPerspective(padded1, warped1, H, padded1.size());

    //add the two images together
    Mat stitched = Mat::zeros(padded1.size(), CV_8UC3);
    warped1.copyTo(stitched);
    for (int y = 0; y < padded2.rows; y++)
    {
        for (int x = 0; x < padded2.cols; x++)
        {
            if (padded2.at<Vec3b>(y, x) != Vec3b(0, 0, 0))
            {
                stitched.at<Vec3b>(y, x) = padded2.at<Vec3b>(y, x);
            }
        }
    }



    //show the result
    imshow("Stitched core 3", stitched);
    waitKey(0);
    destroyAllWindows();
}



/*
    COMPLETION Class Section -> all required COMPLETION classes implemented here.

*/

void cv::Assignment::completion() {
    //load frames
    VideoStabilizer VS;

    std::vector<Mat> images = VS.loadImages(102, "Input_A4/Frame");
    std::cout <<"{" << images.size() <<"} images loaded successfully" << "\n";
    VS.videoStabilization(images);
    VS.exportImages(images, "Stable", "Input_A4/outImgs");
   

}


std::vector<cv::Mat> cv::VideoStabilizer::loadImages(int numFrames, const String prefix) {
    std::vector<Mat> images;
    for (int i = 0; i < numFrames; i++) {
        std::ostringstream frameNumber;
        frameNumber << std::setfill('0') << std::setw(3) << i;
        String filename = prefix + frameNumber.str() + ".jpg";
        Mat image = imread(filename);
        images.push_back(image);
    }

    std::cout << "...Image loading complete \n";
    return images;
}


void cv:: VideoStabilizer::exportImages(const std::vector<Mat>& images, const String prefix, const String outputFolder) {
    for (int i = 0; i < images.size(); i++) {
        std::ostringstream frameNumber;
        frameNumber << std::setw(3) << std::setfill('0') << i;
        String filename = outputFolder + "/" + prefix + frameNumber.str() + ".png";
        imwrite(filename, images[i]);
    }
}


void cv::VideoStabilizer::generate1DGaussian(double mean, double stddev, int size, std::vector<double>& gaussian)
{
    // Create the 1D Gaussian kernel
    Mat kernel = getGaussianKernel(size, stddev, CV_64F);
    gaussian.resize(size);
    for (int i = 0; i < size; ++i)
    {
        gaussian[i] = kernel.at<double>(i);
    }
}

void cv::VideoStabilizer::videoStabilization(std::vector<Mat>& frames)
{
    std::cout << "Initializing video stabilizing... \n";
    // Generate vector of matrix differences between frames (movement performed between each frame)
    std::vector<Mat> differences(frames.size() - 1);
    for (int i = 0; i < frames.size() - 1; ++i)
    {
        differences[i] = frames[i + 1] - frames[i];
    }
    std::cout << "--> vs-computing analysis step 1 complete: Movement performed between each frame\n";

    // Generate vector of cumulative matrices
    std::vector<Mat> cumulativeMatrices(frames.size());
    cumulativeMatrices[0] = Mat::eye(3, 3, CV_64F);

    for (int i = 1; i < frames.size(); ++i)
    {
        // get the frames 
        Mat initialFrame = frames[i];
        Mat prevFrame = frames[i - 1];

        // extract the h matrix from the frames
        Mat h = extraction(initialFrame, prevFrame);

        // Update to store the transformation matrix
        cumulativeMatrices[i] = cumulativeMatrices[i - 1] * h; // TODO needs fixing
        //std::cout<<"->vs - computing analysis "<<i<<"\n";
    }
    std::cout << "--> vs-computing analysis step 2 complete: Generate vector of cumulative matrices\n";

    // Generate 1D Gaussian filter
    double mean = 0.0;
    double stddev = 9;
    int filterSize = 5;  // Window size for the Gaussian filter
    std::vector<double> gaussian;
    generate1DGaussian(mean, stddev, filterSize, gaussian);
    std::cout << "--> vs-computing analysis step 3 complete: Generate 1D Gaussian filter\n";

    // Create vector of cumulative matrices with Gaussian smoothing
    std::vector<Mat> smoothedMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i)
    {
        smoothedMatrices[i] = Mat::zeros(3, 3, CV_64F); //g transformation matrix
        for (int j = -filterSize / 2; j <= filterSize / 2; ++j)
        {
            int index = i + j;
            if (index >= 0 && index < frames.size())
            {
                double weight = gaussian[j + filterSize / 2];
                smoothedMatrices[i] += weight * cumulativeMatrices[index];
            }
        }
    }
    std::cout << "--> vs-computing analysis step 4 complete: Create vector of cumulative matrices with Gaussian smoothing\n";

    // Create vector of stabilization matrices by getting movement and subtracting smooth
    std::vector<Mat> stabilizationMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i)
    {
        stabilizationMatrices[i] = smoothedMatrices[i].inv() * cumulativeMatrices[i];
    }
    std::cout << "--> vs-computing analysis step 5 complete: Create vector of stabilization matrices by getting movement and subtracting smooth\n";

    // Apply stabilization transforms to each frame
    for (int i = 0; i < frames.size(); ++i)
    {
        Mat stabilizedFrame;
        warpPerspective(frames[i], stabilizedFrame, stabilizationMatrices[i], frames[i].size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 0));
        frames[i] = stabilizedFrame;
    }

    std::cout << "... Video successfully stabilized \n";
}


cv::Mat cv::VideoStabilizer::extraction(Mat img_1, Mat img_2) {
    
    // Detect keypoints and compute descriptors using ORB
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    Ptr<ORB> sift = ORB::create();
    sift->detectAndCompute(img_1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img_2, noArray(), keypoints2, descriptors2);

    // Match the descriptors using Brute-Force Matcher
    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Find the homography matrix using RANSAC
    std::vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    Mat H = findHomography(points1, points2, RANSAC);
    //std::cout << "...Extraction complete\n";
    return H;

}

/*
    CHALLENGE Class Section -> all required CHALLENGE classes implemented here.

*/

void cv::Assignment::challenge() {
    //load frames
    VideoStabilizer VS;

    std::vector<Mat> images = VS.loadImages(102, "Input_A4/Frame");
    std::cout << "{" << images.size() << "} images loaded successfully" << "\n";
    VS.minCrop(images);
  

}


void cv::VideoStabilizer::minCrop(std::vector<Mat>& frames){
    int numFrames = frames.size();
    Mat cumulativeMask = Mat::ones(frames[0].size(), CV_8U);

    std::vector<Mat> stabilizationMatrices = stabilizedExtraction(frames);
    for (int i = 0; i < numFrames; ++i)
    {
        Mat mask = Mat::ones(frames[i].size(), CV_8U);
        warpPerspective(mask, mask, stabilizationMatrices[i], frames[i].size());
        cumulativeMask &= mask;
    }

    // Find the bounding box of the non-zero pixels in the cumulative mask
    std::vector<Point> points;
    findNonZero(cumulativeMask, points);
    Rect boundingBox = boundingRect(points);

    // Crop all frames to the bounding box
    
    for (int i = 0; i < numFrames; ++i)
    {
        frames[i] = frames[i](boundingBox);
    }

    //export
    exportImages(frames, "Crop", "Input_A4/croppedImgs");
}

std::vector<cv::Mat> cv::VideoStabilizer::stabilizedExtraction(std::vector<Mat>& frames)
{
    std::cout << "Initializing video stabilizing... \n";
    // Generate vector of matrix differences between frames (movement performed between each frame)
    std::vector<Mat> differences(frames.size() - 1);
    for (int i = 0; i < frames.size() - 1; ++i)
    {
        differences[i] = frames[i + 1] - frames[i];
    }
    std::cout << "--> vs-computing analysis step 1 complete: Movement performed between each frame\n";

    // Generate vector of cumulative matrices
    std::vector<Mat> cumulativeMatrices(frames.size());
    cumulativeMatrices[0] = Mat::eye(3, 3, CV_64F);

    for (int i = 1; i < frames.size(); ++i)
    {
        // get the frames 
        Mat initialFrame = frames[i];
        Mat prevFrame = frames[i - 1];

        // extract the h matrix from the frames
        Mat h = extraction(initialFrame, prevFrame);

        // Update to store the transformation matrix
        cumulativeMatrices[i] = cumulativeMatrices[i - 1] * h; // TODO needs fixing
        //std::cout<<"->vs - computing analysis "<<i<<"\n";
    }
    std::cout << "--> vs-computing analysis step 2 complete: Generate vector of cumulative matrices\n";

    // Generate 1D Gaussian filter
    double mean = 0.0;
    double stddev = 5;
    int filterSize = 9;  // Window size for the Gaussian filter
    std::vector<double> gaussian;
    generate1DGaussian(mean, stddev, filterSize, gaussian);
    std::cout << "--> vs-computing analysis step 3 complete: Generate 1D Gaussian filter\n";

    // Create vector of cumulative matrices with Gaussian smoothing
    std::vector<Mat> smoothedMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i)
    {
        smoothedMatrices[i] = Mat::zeros(3, 3, CV_64F); //g transformation matrix
        for (int j = -filterSize / 2; j <= filterSize / 2; ++j)
        {
            int index = i + j;
            if (index >= 0 && index < frames.size())
            {
                double weight = gaussian[j + filterSize / 2];
                smoothedMatrices[i] += weight * cumulativeMatrices[index];
            }
        }
    }
    std::cout << "--> vs-computing analysis step 4 complete: Create vector of cumulative matrices with Gaussian smoothing\n";

    // Create vector of stabilization matrices by getting movement and subtracting smooth
    std::vector<Mat> stabilizationMatrices(frames.size());
    for (int i = 0; i < frames.size(); ++i)
    {
        stabilizationMatrices[i] = smoothedMatrices[i].inv() * cumulativeMatrices[i];
    }

    // Apply stabilization transforms to each frame
    for (int i = 0; i < frames.size(); ++i)
    {
        Mat stabilizedFrame;
        warpPerspective(frames[i], stabilizedFrame, stabilizationMatrices[i], frames[i].size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 0));
        frames[i] = stabilizedFrame;
    }

    return  stabilizationMatrices;
    std::cout << "... Video successfully stabilized \n";
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


