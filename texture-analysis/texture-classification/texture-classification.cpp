#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <map>
#include <opencv2/opencv.hpp>

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
            Mat filter2D = kernels1D[i] * kernels1D[j].t();
            filters25.push_back(filter2D);
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

void analyzeDiscriminantPower(const Mat& trainFeatures, const vector<int>& trainLabels, int numClasses) {
    int numFeatures = trainFeatures.cols;
    vector<float> fisherRatios(numFeatures, 0.0f);
    Mat globalMeans;
    cv::reduce(trainFeatures, globalMeans, 0, cv::REDUCE_AVG);

    for (int f = 0; f < numFeatures; f++) {
        float betweenClassVariance = 0.0f;
        float withinClassVariance = 0.0f;

        for (int c = 0; c < numClasses; c++) {
            Mat classFeatures;
            for (int r = 0; r < trainFeatures.rows; r++) {
                if (trainLabels[r] == c) classFeatures.push_back(trainFeatures.row(r));
            }
            if (classFeatures.empty()) continue;

            Mat classMean, classStdDev;
            meanStdDev(classFeatures.col(f), classMean, classStdDev);

            float meanVal = classMean.at<double>(0, 0);
            float varVal = pow(classStdDev.at<double>(0, 0), 2);
            int count = classFeatures.rows;
            float globalMeanVal = globalMeans.at<float>(0, f);

            betweenClassVariance += count * pow(meanVal - globalMeanVal, 2);
            withinClassVariance += count * varVal;
        }
        fisherRatios[f] = betweenClassVariance / (withinClassVariance + 1e-6);
    }
    auto maxIt = max_element(fisherRatios.begin(), fisherRatios.end());
    auto minIt = min_element(fisherRatios.begin(), fisherRatios.end());

    cout << "Strongest Dimension: " << distance(fisherRatios.begin(), maxIt) << ", Ratio: " << *maxIt << endl;
    cout << "Weakest Dimension: " << distance(fisherRatios.begin(), minIt) << ", Ratio: " << *minIt << endl;
}

void saveFeaturesToCSV(const string& filename, const Mat& features, const vector<int>& labels) {
    ofstream file(filename);
    file << "PC1,PC2,PC3,Label\n";
    for (int i = 0; i < features.rows; i++) {
        file << features.at<float>(i, 0) << "," << features.at<float>(i, 1) << "," << features.at<float>(i, 2) << "," << labels[i] << "\n";
    }
    file.close();
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

    vector<Mat> filters = generateLawsFilters();
    Mat trainFeatures(trainFiles.size(), 25, CV_32F);
    Mat testFeatures(testFiles.size(), 25, CV_32F);

    for (size_t i = 0; i < trainFiles.size(); i++) {
        Mat img = readRawImage(trainFiles[i]);
        vector<float> feats = extractFeatures(img, filters);
        for(int j=0; j<25; j++) trainFeatures.at<float>(i, j) = feats[j];
    }
    for (size_t i = 0; i < testFiles.size(); i++) {
        Mat img = readRawImage(testFiles[i]);
        vector<float> feats = extractFeatures(img, filters);
        for(int j=0; j<25; j++) testFeatures.at<float>(i, j) = feats[j];
    }

    analyzeDiscriminantPower(trainFeatures, trainLabels, 4);

    PCA pca(trainFeatures, Mat(), PCA::DATA_AS_ROW, 3);
    Mat trainReduced = pca.project(trainFeatures);
    Mat testReduced = pca.project(testFeatures);

    saveFeaturesToCSV("train_3d_features.csv", trainReduced, trainLabels);

    Mat covar, mean_val;
    calcCovarMatrix(trainReduced, covar, mean_val, COVAR_NORMAL | COVAR_ROWS, CV_32F);
    covar = covar / (trainReduced.rows - 1);
    
    Mat icovar;
    invert(covar, icovar, DECOMP_SVD);
    int errors = 0;

    for (int i = 0; i < testReduced.rows; i++) {
        double minDistance = DBL_MAX;
        int predictedLabel = -1;
        
        Mat testSample = testReduced.row(i);

        for (int j = 0; j < trainReduced.rows; j++) {
            Mat trainSample = trainReduced.row(j);
            double dist = Mahalanobis(testSample, trainSample, icovar);
            
            if (dist < minDistance) {
                minDistance = dist;
                predictedLabel = trainLabels[j];
            }
        }

        cout << "Test file: " << i+1 << ".raw | True: " << classNames[testLabels[i]] << "| Pred: " << classNames[predictedLabel] << endl;
             
        if (predictedLabel != testLabels[i]) {
            errors++;
        }
    }

    float errorRate = (float)errors / testReduced.rows * 100.0f;
    cout << "Total: " << testReduced.rows << endl;
    cout << "Misclassifications: " << errors << endl;
    cout << "Test Error Rate: " << errorRate << "%" << endl;

    return 0;
}