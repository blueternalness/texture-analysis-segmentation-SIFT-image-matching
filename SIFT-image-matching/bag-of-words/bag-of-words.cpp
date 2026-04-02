#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

// Helper function to read the 600x400 RGB .raw files
Mat readRawRGB(const string& filename, int width = 600, int height = 400) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << filename << endl;
        exit(1);
    }
    
    Mat img(height, width, CV_8UC3);
    file.read(reinterpret_cast<char*>(img.data), width * height * 3);
    file.close();
    
    // Convert RGB (raw format) to BGR (OpenCV standard)
    cvtColor(img, img, COLOR_RGB2BGR);
    return img;
}

// Helper to print the 8-bin histogram nicely
void printHistogram(const string& name, const Mat& hist) {
    cout << name << " BoW Histogram (8 bins): [ ";
    for (int i = 0; i < hist.cols; i++) {
        cout << fixed << setprecision(4) << hist.at<float>(0, i) << " ";
    }
    cout << "]" << endl;
}

int main() {
    int w = 600, h = 400;

    // 1. Read Images
    Mat cat1 = readRawRGB("Cat_1.raw", w, h);
    Mat cat2 = readRawRGB("Cat_2.raw", w, h);
    Mat cat3 = readRawRGB("Cat_3.raw", w, h);
    Mat dog1 = readRawRGB("Dog_1.raw", w, h);

    // 2. Initialize SIFT Extractor
    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> kp_c1, kp_c2, kp_c3, kp_d1;
    Mat desc_c1, desc_c2, desc_c3, desc_d1;

    sift->detectAndCompute(cat1, noArray(), kp_c1, desc_c1);
    sift->detectAndCompute(cat2, noArray(), kp_c2, desc_c2);
    sift->detectAndCompute(cat3, noArray(), kp_c3, desc_c3);
    sift->detectAndCompute(dog1, noArray(), kp_d1, desc_d1);

    // 3. Pool all descriptors together to build the vocabulary
    Mat all_descriptors;
    all_descriptors.push_back(desc_c1);
    all_descriptors.push_back(desc_c2);
    all_descriptors.push_back(desc_c3);
    all_descriptors.push_back(desc_d1);

    cout << "Total SIFT descriptors extracted: " << all_descriptors.rows << endl;

    // 4. Run K-Means Clustering to create the Codebook (k=8)
    int dictionarySize = 8;
    TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001);
    int retries = 1;
    int flags = KMEANS_PP_CENTERS;
    
    cout << "Clustering SIFT features into " << dictionarySize << " visual words..." << endl;
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
    bowTrainer.add(all_descriptors);
    Mat vocabulary = bowTrainer.cluster();

    // 5. Initialize the Bag of Words Descriptor Extractor
    // FlannBasedMatcher is efficient for floating-point SIFT descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    BOWImgDescriptorExtractor bowDE(sift, matcher);
    bowDE.setVocabulary(vocabulary);

    // 6. Compute BoW Histograms for each image
    Mat bow_cat1, bow_cat2, bow_cat3, bow_dog1;
    bowDE.compute(cat1, kp_c1, bow_cat1);
    bowDE.compute(cat2, kp_c2, bow_cat2);
    bowDE.compute(cat3, kp_c3, bow_cat3);
    bowDE.compute(dog1, kp_d1, bow_dog1);

    cout << "\n--- Bag of Words Histograms ---" << endl;
    printHistogram("Cat_1", bow_cat1);
    printHistogram("Cat_2", bow_cat2);
    printHistogram("Cat_3", bow_cat3);
    printHistogram("Dog_1", bow_dog1);

    // 7. Match Cat_3's codewords with the others using L2 Distance
    // Lower distance = higher similarity
    double dist_c3_c1 = norm(bow_cat3, bow_cat1, NORM_L2);
    double dist_c3_c2 = norm(bow_cat3, bow_cat2, NORM_L2);
    double dist_c3_d1 = norm(bow_cat3, bow_dog1, NORM_L2);

    cout << "\n--- Matching Results (L2 Distance relative to Cat_3) ---" << endl;
    cout << "Cat_3 vs Cat_1: " << dist_c3_c1 << endl;
    cout << "Cat_3 vs Cat_2: " << dist_c3_c2 << endl;
    cout << "Cat_3 vs Dog_1: " << dist_c3_d1 << endl;
    
    cout << "\nNote: Lower distance indicates a closer match in the Bag-of-Words space." << endl;

    return 0;
}