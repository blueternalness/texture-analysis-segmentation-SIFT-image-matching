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

// -------------------------------------------------------------------------
// THE MANUAL BAG-OF-WORDS ENGINE
// -------------------------------------------------------------------------
// This function takes a single image's descriptors and maps them to the 
// nearest visual word in our K-Means vocabulary to create a histogram.
Mat computeManualBoW(const Mat& image_descriptors, const Mat& vocabulary, int k) {
    // Initialize an empty 1-row, k-column matrix (all zeros) for the histogram
    Mat histogram = Mat::zeros(1, k, CV_32F);

    if (image_descriptors.empty()) return histogram;

    // 1. For every single SIFT descriptor in this specific image...
    for (int i = 0; i < image_descriptors.rows; i++) {
        Mat single_descriptor = image_descriptors.row(i);
        single_descriptor.convertTo(single_descriptor, CV_32F); // Ensure float32

        // 2. Find the nearest "word" (cluster center) in the vocabulary
        int best_word_idx = -1;
        float min_distance = FLT_MAX;

        for (int j = 0; j < vocabulary.rows; j++) {
            Mat word = vocabulary.row(j);
            // Calculate Euclidean (L2) distance between the descriptor and the word
            float dist = norm(single_descriptor, word, NORM_L2);

            if (dist < min_distance) {
                min_distance = dist;
                best_word_idx = j;
            }
        }

        // 3. Add a tally to the winning word's bin
        histogram.at<float>(0, best_word_idx) += 1.0f;
    }

    // 4. L1 Normalization (convert raw counts to percentages)
    // This allows us to compare images that have different numbers of keypoints.
    histogram /= (float)image_descriptors.rows;

    return histogram;
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

    // 2. Initialize Base SIFT Extractor
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

    // Standardize matrix type for K-Means (requires CV_32F)
    Mat all_desc_32f;
    all_descriptors.convertTo(all_desc_32f, CV_32F);

    cout << "Total SIFT descriptors extracted: " << all_desc_32f.rows << endl;

    // 4. Run Standard K-Means Clustering
    int k = 8;
    Mat labels, vocabulary;
    TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.001);
    int attempts = 3; 
    int flags = KMEANS_PP_CENTERS; 
    
    cout << "Clustering SIFT features into " << k << " visual words using cv::kmeans..." << endl;
    kmeans(all_desc_32f, k, labels, tc, attempts, flags, vocabulary);

    // 5. Compute Histograms Manually
    Mat bow_cat1 = computeManualBoW(desc_c1, vocabulary, k);
    Mat bow_cat2 = computeManualBoW(desc_c2, vocabulary, k);
    Mat bow_cat3 = computeManualBoW(desc_c3, vocabulary, k);
    Mat bow_dog1 = computeManualBoW(desc_d1, vocabulary, k);

    cout << "\n--- Bag of Words Histograms ---" << endl;
    printHistogram("Cat_1", bow_cat1);
    printHistogram("Cat_2", bow_cat2);
    printHistogram("Cat_3", bow_cat3);
    printHistogram("Dog_1", bow_dog1);

    // 6. Match Cat_3's codewords with the others using L2 Distance
    double dist_c3_c1 = norm(bow_cat3, bow_cat1, NORM_L2);
    double dist_c3_c2 = norm(bow_cat3, bow_cat2, NORM_L2);
    double dist_c3_d1 = norm(bow_cat3, bow_dog1, NORM_L2);

    cout << "\n--- Matching Results (L2 Distance relative to Cat_3) ---" << endl;
    cout << "Cat_3 vs Cat_1: " << fixed << setprecision(4) << dist_c3_c1 << endl;
    cout << "Cat_3 vs Cat_2: " << fixed << setprecision(4) << dist_c3_c2 << endl;
    cout << "Cat_3 vs Dog_1: " << fixed << setprecision(4) << dist_c3_d1 << endl;
    
    cout << "\nNote: Lower distance indicates a closer match in the Bag-of-Words space." << endl;

    return 0;
}