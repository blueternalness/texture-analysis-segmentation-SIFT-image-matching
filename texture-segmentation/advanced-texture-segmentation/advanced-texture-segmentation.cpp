#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <opencv2/opencv.hpp> // ONLY used for cv::PCA

using namespace std;

const int WIDTH = 512;
const int HEIGHT = 512;
const int NUM_FILTERS = 25;
const int K = 6;              
const int WINDOW_SIZE = 15;   
const int SMALL_WINDOW = 5;   // Used for boundary refinement
const int PCA_DIMS = 5;       

// 1D Laws Vectors
const double laws1D[5][5] = {
    { 1,  4,  6,  4,  1}, // L5 
    {-1, -2,  0,  2,  1}, // E5 
    {-1,  0,  2,  0, -1}, // S5 
    {-1,  2,  0, -2,  1}, // W5 
    { 1, -4,  6, -4,  1}  // R5 
};

// --- Standard C++ Helpers ---
int mirrorBoundary(int val, int max_val) {
    if (val < 0) return -val;
    if (val >= max_val) return 2 * max_val - 2 - val;
    return val;
}

vector<double> readRawImage(const string& filename) {
    vector<unsigned char> buffer(WIDTH * HEIGHT);
    ifstream file(filename, ios::binary);
    if (!file) { cerr << "Error opening " << filename << endl; exit(1); }
    file.read(reinterpret_cast<char*>(buffer.data()), WIDTH * HEIGHT);
    return vector<double>(buffer.begin(), buffer.end());
}

void writeRawImage(const string& filename, const vector<unsigned char>& img) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(img.data()), WIDTH * HEIGHT);
}

// --- Standard C++ Filtering & Energy ---
vector<double> convolve2D(const vector<double>& img, const vector<double>& filter) {
    vector<double> result(WIDTH * HEIGHT, 0.0);
    int offset = 2;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sum = 0.0;
            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    int ny = mirrorBoundary(y + fy, HEIGHT);
                    int nx = mirrorBoundary(x + fx, WIDTH);
                    sum += img[ny * WIDTH + nx] * filter[(fy + offset) * 5 + (fx + offset)];
                }
            }
            result[y * WIDTH + x] = sum;
        }
    }
    return result;
}

vector<double> computeEnergy(const vector<double>& response, int w_size) {
    vector<double> energy(WIDTH * HEIGHT, 0.0);
    int offset = w_size / 2;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sum = 0.0;
            for (int wy = -offset; wy <= offset; ++wy) {
                for (int wx = -offset; wx <= offset; ++wx) {
                    int ny = mirrorBoundary(y + wy, HEIGHT);
                    int nx = mirrorBoundary(x + wx, WIDTH);
                    sum += abs(response[ny * WIDTH + nx]);
                }
            }
            energy[y * WIDTH + x] = sum / (w_size * w_size);
        }
    }
    return energy;
}

// Struct to hold K-Means results (labels and centroids)
struct KMeansResult {
    vector<int> labels;
    vector<vector<double>> centroids;
};

// --- Standard C++ K-Means ---
KMeansResult kmeans(const vector<vector<double>>& features, int num_clusters) {
    int num_pixels = features.size();
    int num_features = features[0].size();
    srand(42); 
    
    vector<vector<double>> centroids(num_clusters, vector<double>(num_features));
    for (int k = 0; k < num_clusters; ++k) centroids[k] = features[rand() % num_pixels];

    vector<int> labels(num_pixels, 0);
    bool changed = true;
    int iter = 0;

    while (changed && iter < 30) {
        changed = false;
        for (int i = 0; i < num_pixels; ++i) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;
            for (int k = 0; k < num_clusters; ++k) {
                double dist = 0.0;
                for (int f = 0; f < num_features; ++f) {
                    double diff = features[i][f] - centroids[k][f];
                    dist += diff * diff;
                }
                if (dist < min_dist) { min_dist = dist; best_cluster = k; }
            }
            if (labels[i] != best_cluster) { labels[i] = best_cluster; changed = true; }
        }

        vector<vector<double>> new_centroids(num_clusters, vector<double>(num_features, 0.0));
        vector<int> counts(num_clusters, 0);
        for (int i = 0; i < num_pixels; ++i) {
            counts[labels[i]]++;
            for (int f = 0; f < num_features; ++f) new_centroids[labels[i]][f] += features[i][f];
        }
        for (int k = 0; k < num_clusters; ++k) {
            if (counts[k] > 0) {
                for (int f = 0; f < num_features; ++f) centroids[k][f] = new_centroids[k][f] / counts[k];
            }
        }
    }
    return {labels, centroids};
}

// --- Standard C++ Majority Filter ---
vector<int> majorityFilter(const vector<int>& labels, int w_size) {
    vector<int> filtered = labels;
    int offset = w_size / 2;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            vector<int> counts(K, 0);
            for (int wy = -offset; wy <= offset; ++wy) {
                for (int wx = -offset; wx <= offset; ++wx) {
                    int ny = mirrorBoundary(y + wy, HEIGHT);
                    int nx = mirrorBoundary(x + wx, WIDTH);
                    counts[labels[ny * WIDTH + nx]]++;
                }
            }
            int max_count = 0, best_label = labels[y * WIDTH + x];
            for (int k = 0; k < K; k++) {
                if (counts[k] > max_count) { max_count = counts[k]; best_label = k; }
            }
            filtered[y * WIDTH + x] = best_label;
        }
    }
    return filtered;
}

// --- Main Pipeline ---
int main() {
    cout << "Reading image..." << endl;
    vector<double> img = readRawImage("Mosaic.raw");

    // 1. Generate Filter Bank
    vector<vector<double>> filters(NUM_FILTERS, vector<double>(25));
    int f_idx = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) filters[f_idx][y * 5 + x] = laws1D[i][y] * laws1D[j][x];
            }
            f_idx++;
        }
    }

    // 2. Convolution & Energy
    cout << "Computing responses & macro-energy (15x15)..." << endl;
    vector<vector<double>> responses(NUM_FILTERS);
    vector<vector<double>> energies(NUM_FILTERS);
    for (int i = 0; i < NUM_FILTERS; ++i) {
        responses[i] = convolve2D(img, filters[i]);
        energies[i] = computeEnergy(responses[i], WINDOW_SIZE);
    }

    // 3. Normalization (Discard L5L5)
    cout << "Normalizing 24D features..." << endl;
    int num_pixels = WIDTH * HEIGHT;
    vector<vector<double>> features(num_pixels, vector<double>(24));
    for (int p = 0; p < num_pixels; ++p) {
        double L5L5 = max(energies[0][p], 1e-5); 
        for (int i = 1; i < NUM_FILTERS; ++i) features[p][i - 1] = energies[i][p] / L5L5;
    }

    // 4. OpenCV PCA
    cout << "Running OpenCV PCA..." << endl;
    cv::Mat features_mat(num_pixels, 24, CV_32F);
    for (int i = 0; i < num_pixels; ++i) 
        for (int j = 0; j < 24; ++j) features_mat.at<float>(i, j) = static_cast<float>(features[i][j]);

    cv::PCA pca(features_mat, cv::Mat(), cv::PCA::DATA_AS_ROW, PCA_DIMS);
    cv::Mat reduced_mat = pca.project(features_mat);

    vector<vector<double>> reduced_features(num_pixels, vector<double>(PCA_DIMS));
    for (int i = 0; i < num_pixels; ++i) 
        for (int j = 0; j < PCA_DIMS; ++j) reduced_features[i][j] = static_cast<double>(reduced_mat.at<float>(i, j));

    // 5. Initial Segmentation
    cout << "Running Initial K-Means..." << endl;
    KMeansResult kmeans_res = kmeans(reduced_features, K);
    vector<int> labels = kmeans_res.labels;

    // 6. Advanced: Texture-Based Boundary Refinement
    cout << "Refining boundaries based on micro-texture (5x5)..." << endl;
    
    // a. Compute refined energies with small window
    vector<vector<double>> refined_energies(NUM_FILTERS);
    for (int i = 0; i < NUM_FILTERS; ++i) {
        refined_energies[i] = computeEnergy(responses[i], SMALL_WINDOW);
    }

    // b. Identify boundaries and re-classify
    for (int y = 1; y < HEIGHT - 1; ++y) {
        for (int x = 1; x < WIDTH - 1; ++x) {
            int p_idx = y * WIDTH + x;
            int current_label = labels[p_idx];
            
            // If pixel is on a boundary
            if (labels[(y-1)*WIDTH + x] != current_label || labels[(y+1)*WIDTH + x] != current_label ||
                labels[y*WIDTH + x - 1] != current_label || labels[y*WIDTH + x + 1] != current_label) {
                
                // Form new 24D feature
                cv::Mat pixel_feature(1, 24, CV_32F);
                double L5L5 = max(refined_energies[0][p_idx], 1e-5);
                for (int i = 1; i < NUM_FILTERS; ++i) {
                    pixel_feature.at<float>(0, i - 1) = static_cast<float>(refined_energies[i][p_idx] / L5L5);
                }

                // Project through PCA
                cv::Mat reduced_pixel = pca.project(pixel_feature);

                // Find closest centroid
                double min_dist = numeric_limits<double>::max();
                int best_label = current_label;
                for (int k = 0; k < K; ++k) {
                    double dist = 0.0;
                    for(int f=0; f < PCA_DIMS; ++f) {
                        double diff = reduced_pixel.at<float>(0, f) - kmeans_res.centroids[k][f];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) { min_dist = dist; best_label = k; }
                }
                labels[p_idx] = best_label;
            }
        }
    }

    // 7. Post-Processing: Majority Filter
    cout << "Applying Majority Filter for stray holes..." << endl;
    labels = majorityFilter(labels, 9); 

    // 8. Output
    vector<unsigned char> output_img(num_pixels);
    int step = 255 / (K - 1);
    for (int i = 0; i < num_pixels; ++i) {
        output_img[i] = static_cast<unsigned char>(labels[i] * step);
    }

    cout << "Writing Final_Advanced_Output.raw..." << endl;
    writeRawImage("p2b_output.raw", output_img);
    cout << "Done!" << endl;

    return 0;
}