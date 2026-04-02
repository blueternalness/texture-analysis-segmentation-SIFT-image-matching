#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

Mat readRawImage(const string& filename, int width = 128, int height = 128) {
    Mat image(height, width, CV_8UC1);
    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(image.data), width * height);
    return image;
}

vector<Mat> generateLawsFilters() {
    Mat L5 = (Mat_<float>(5, 1) << 1, 4, 6, 4, 1);
    Mat E5 = (Mat_<float>(5, 1) << -1, -2, 0, 2, 1);
    Mat S5 = (Mat_<float>(5, 1) << -1, 0, 2, 0, -1);
    Mat W5 = (Mat_<float>(5, 1) << -1, 2, 0, -2, 1);
    Mat R5 = (Mat_<float>(5, 1) << 1, -4, 6, -4, 1);
    vector<Mat> kernels1D = {L5, E5, S5, W5, R5};
    vector<Mat> filters25;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            filters25.push_back(kernels1D[i] * kernels1D[j].t());
        }
    }
    return filters25;
}

vector<float> extractFeatures(const Mat& image, const vector<Mat>& filters) {
    Mat imgFloat;
    image.convertTo(imgFloat, CV_32F);
    vector<float> featureVector;
    for (const auto& filter : filters) {
        Mat response;
        filter2D(imgFloat, response, -1, filter, Point(-1, -1), 0, BORDER_REFLECT);
        Mat absResponse = cv::abs(response);
        Scalar meanEnergy = mean(absResponse);
        featureVector.push_back(meanEnergy[0]);
    }
    return featureVector;
}
float evaluateKMeans(const Mat& testFeatures, const vector<int>& testLabels, int K) {
    Mat bestLabels;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0);
    kmeans(testFeatures, K, bestLabels, criteria, 10, KMEANS_PP_CENTERS);
    int errors = 0;
    for (int k = 0; k < K; ++k) {
        map<int, int> classCounts;
        vector<int> pointsInCluster;
        
        for (int i = 0; i < bestLabels.rows; ++i) {
            if (bestLabels.at<int>(i, 0) == k) {
                classCounts[testLabels[i]]++;
                pointsInCluster.push_back(i);
            }
        }
        
        int majorityClass = -1;
        int maxCount = -1;
        for (auto const& [cls, count] : classCounts) {
            if (count > maxCount) {
                maxCount = count;
                majorityClass = cls;
            }
        }
        for (int idx : pointsInCluster) {
            if (testLabels[idx] != majorityClass) {
                errors++;
            }
        }
    }
    
    return (float)errors / testLabels.size() * 100.0f;
}

float evaluateSVM(const Mat& trainFeat, const vector<int>& trainLbls, const Mat& testFeat, const vector<int>& testLbls) {
    Mat trainLabelsMat(trainLbls.size(), 1, CV_32SC1, (void*)trainLbls.data());
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainFeat, ml::ROW_SAMPLE, trainLabelsMat);
    
    int errors = 0;
    for (int i = 0; i < testFeat.rows; ++i) {
        float pred = svm->predict(testFeat.row(i));
        if ((int)pred != testLbls[i]) {
            errors++;
        }
    }
    return (float)errors / testLbls.size() * 100.0f;
}


int main() {
    vector<string> classNames = {"blanket", "brick", "grass", "stones"};
    string trainDir = "train/";
    string testDir = "test/";
    vector<string> trainFiles;
    vector<int> trainLabels;
    
    for (int c = 0; c < classNames.size(); c++) {
        for (int i = 1; i <= 9; i++) {
            trainFiles.push_back(trainDir + classNames[c] + "_" + to_string(i) + ".raw");
            trainLabels.push_back(c);
        }
    }

    vector<string> testFiles;
    vector<int> testLabels;
    
    string labelFile = testDir + "test_label.txt";
    ifstream infile(labelFile);
    if (!infile.is_open()) {
        cerr << "\n[WARNING] Could not open " << labelFile << ". Using hardcoded test labels." << endl;
        int fallback[] = {2, 0, 0, 3, 3, 2, 1, 3, 1, 1, 0, 2}; 
        for (int i = 1; i <= 12; i++) {
            testFiles.push_back(testDir + to_string(i) + ".raw");
            testLabels.push_back(fallback[i-1]);
        }
    } else {
        string line;
        int index = 1;
        while (getline(infile, line) && index <= 12) {
            testFiles.push_back(testDir + to_string(index) + ".raw");
            int labelID = -1;
            if (line.find("Blanket") != string::npos) labelID = 0;
            else if (line.find("Brick") != string::npos) labelID = 1;
            else if (line.find("Grass") != string::npos) labelID = 2;
            else if (line.find("Stones") != string::npos) labelID = 3;
            testLabels.push_back(labelID);
            index++;
        }
    }
    vector<Mat> filters = generateLawsFilters();
    Mat trainFeat25D(trainFiles.size(), 25, CV_32F);
    Mat testFeat25D(testFiles.size(), 25, CV_32F);

    for (size_t i = 0; i < trainFiles.size(); i++) {
        Mat img = readRawImage(trainFiles[i]);
        vector<float> feats = extractFeatures(img, filters);
        for(int j=0; j<25; j++){
            trainFeat25D.at<float>(i, j) = feats[j];
        }
    }
    for (size_t i = 0; i < testFiles.size(); i++) {
        Mat img = readRawImage(testFiles[i]);
        vector<float> feats = extractFeatures(img, filters);
        for(int j=0; j<25; j++){
            testFeat25D.at<float>(i, j) = feats[j];
        }
    }
    PCA pca(trainFeat25D, Mat(), PCA::DATA_AS_ROW, 3);
    Mat trainFeat3D = pca.project(trainFeat25D);
    Mat testFeat3D = pca.project(testFeat25D);

    int K = 4;
    float kmeansError25D = evaluateKMeans(testFeat25D, testLabels, K);
    float kmeansError3D = evaluateKMeans(testFeat3D, testLabels, K);
    
    cout << "K-Means Error(25-D): " << kmeansError25D << "%" << endl;
    cout << "K-Means Error(3-D): " << kmeansError3D << "%" << endl;
    
    float svmError25D = evaluateSVM(trainFeat25D, trainLabels, testFeat25D, testLabels);
    float svmError3D = evaluateSVM(trainFeat3D, trainLabels, testFeat3D, testLabels);
    cout << "SVM Error (25-D): " << svmError25D << "%" << endl;
    cout << "SVM Error (3-D): " << svmError3D << "%" << endl;
    return 0;
}