#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

Mat readRawRGB(const string& filename, int width = 600, int height = 400) {
    ifstream file(filename, ios::binary);
    Mat img(height, width, CV_8UC3);
    file.read(reinterpret_cast<char*>(img.data), width * height * 3);
    file.close();
    
    cvtColor(img, img, COLOR_RGB2BGR);
    return img;
}

Mat computeBoW(const Mat& image_descriptors, const Mat& vocabulary, int k) {
    Mat histogram = Mat::zeros(1, k, CV_32F);
    if (image_descriptors.empty()) return histogram;

    for (int i = 0; i < image_descriptors.rows; i++) {
        Mat single_descriptor = image_descriptors.row(i);
        single_descriptor.convertTo(single_descriptor, CV_32F);
        int best_word_idx = -1;
        float min_distance = FLT_MAX;

        for (int j = 0; j < vocabulary.rows; j++) {
            Mat word = vocabulary.row(j);
            float dist = norm(single_descriptor, word, NORM_L2);

            if (dist < min_distance) {
                min_distance = dist;
                best_word_idx = j;
            }
        }
        histogram.at<float>(0, best_word_idx) += 1.0f;
    }
    histogram /= (float)image_descriptors.rows;

    return histogram;
}

void printHistogram(const string& name, const Mat& hist) {
    cout << name << "Histogram: ";
    for (int i = 0; i < hist.cols; i++) {
        cout << fixed << setprecision(4) << hist.at<float>(0, i) << " ";
    }
    cout << endl;
}

int main() {
    int w = 600, h = 400;
    Mat cat1 = readRawRGB("Cat_1.raw", w, h);
    Mat cat2 = readRawRGB("Cat_2.raw", w, h);
    Mat cat3 = readRawRGB("Cat_3.raw", w, h);
    Mat dog1 = readRawRGB("Dog_1.raw", w, h);
    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> kp_c1, kp_c2, kp_c3, kp_d1;
    Mat desc_c1, desc_c2, desc_c3, desc_d1;

    sift->detectAndCompute(cat1, noArray(), kp_c1, desc_c1);
    sift->detectAndCompute(cat2, noArray(), kp_c2, desc_c2);
    sift->detectAndCompute(cat3, noArray(), kp_c3, desc_c3);
    sift->detectAndCompute(dog1, noArray(), kp_d1, desc_d1);

    Mat all_descriptors;
    all_descriptors.push_back(desc_c1);
    all_descriptors.push_back(desc_c2);
    all_descriptors.push_back(desc_c3);
    all_descriptors.push_back(desc_d1);

    Mat all_desc_32f;
    all_descriptors.convertTo(all_desc_32f, CV_32F);

    cout << "# of SIFT descriptors: " << all_desc_32f.rows << endl;
    int k = 8;
    Mat labels, vocabulary;
    TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001);
    int attempts = 3; 
    int flags = KMEANS_PP_CENTERS; 
    kmeans(all_desc_32f, k, labels, tc, attempts, flags, vocabulary);

    Mat bow_cat1 = computeBoW(desc_c1, vocabulary, k);
    Mat bow_cat2 = computeBoW(desc_c2, vocabulary, k);
    Mat bow_cat3 = computeBoW(desc_c3, vocabulary, k);
    Mat bow_dog1 = computeBoW(desc_d1, vocabulary, k);

    printHistogram("Cat_1", bow_cat1);
    printHistogram("Cat_2", bow_cat2);
    printHistogram("Cat_3", bow_cat3);
    printHistogram("Dog_1", bow_dog1);
    double dist_c3_c1 = norm(bow_cat3, bow_cat1, NORM_L2);
    double dist_c3_c2 = norm(bow_cat3, bow_cat2, NORM_L2);
    double dist_c3_d1 = norm(bow_cat3, bow_dog1, NORM_L2);

    cout << "Cat_3 vs Cat_1: " << fixed << setprecision(4) << dist_c3_c1 << endl;
    cout << "Cat_3 vs Cat_2: " << fixed << setprecision(4) << dist_c3_c2 << endl;
    cout << "Cat_3 vs Dog_1: " << fixed << setprecision(4) << dist_c3_d1 << endl;
    return 0;
}