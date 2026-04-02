#include <fstream>
#include <vector>
#include <string>
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

void matchAndSave(const string& name1, const Mat& img1, const vector<KeyPoint>& kp1, const Mat& desc1, const string& name2, const Mat& img2, const vector<KeyPoint>& kp2, const Mat& desc2, const string& outputFilename) {                      
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);
    const float ratio_thresh = 0.75f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    Mat img_matches;
    drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite(outputFilename, img_matches);
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

    cout << "Keypoints:" << endl;
    cout << "Cat_1: " << kp_c1.size() << " keypoints" << endl;
    cout << "Cat_3: " << kp_c3.size() << " keypoints" << endl;

    int largest_idx = 0;
    float max_scale = 0.0f;
    for (size_t i = 0; i < kp_c1.size(); i++) {
        if (kp_c1[i].size > max_scale) {
            max_scale = kp_c1[i].size;
            largest_idx = i;
        }
    }
    Mat single_desc = desc_c1.row(largest_idx);
    BFMatcher single_matcher(NORM_L2);
    vector<DMatch> single_match;
    single_matcher.match(single_desc, desc_c3, single_match);
    int matched_idx_in_c3 = single_match[0].trainIdx;

    cout << "Largest keypoint in Cat_1:" << endl;
    cout << "  Scale: " << kp_c1[largest_idx].size << endl;
    cout << "  Orientation: " << kp_c1[largest_idx].angle << endl;
    
    cout << "Closest neighbor in Cat_3:" << endl;
    cout << "  Scale: " << kp_c3[matched_idx_in_c3].size << endl;
    cout << "  Orientation: " << kp_c3[matched_idx_in_c3].angle << endl;

    matchAndSave("Cat_1", cat1, kp_c1, desc_c1, "Cat_3", cat3, kp_c3, desc_c3, "Match_Cat1_Cat3.jpg");
    matchAndSave("Cat_3", cat3, kp_c3, desc_c3, "Cat_2", cat2, kp_c2, desc_c2, "Match_Cat3_Cat2.jpg");
    matchAndSave("Dog_1", dog1, kp_d1, desc_d1, "Cat_3", cat3, kp_c3, desc_c3, "Match_Dog1_Cat3.jpg");
    matchAndSave("Cat_1", cat1, kp_c1, desc_c1, "Dog_1", dog1, kp_d1, desc_d1, "Match_Cat1_Dog1.jpg");

    return 0;
}