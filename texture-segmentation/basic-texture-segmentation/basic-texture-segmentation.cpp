#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
using namespace std;
const int WIDTH = 512;
const int HEIGHT = 512;
const int NUM_FILTERS = 25;
const int K = 6;
const int WINDOW_SIZE = 15;

const double laws1D[5][5] = {
    { 1,  4,  6,  4,  1}, //L5
    {-1, -2,  0,  2,  1}, //E5
    {-1,  0,  2,  0, -1}, //S5
    {-1,  2,  0, -2,  1}, //W5
    { 1, -4,  6, -4,  1}  //R5
};
int mirrorBoundary(int val, int max_val) {
    if (val < 0){
        return -val;
    }
    if (val >= max_val){
        return 2 * max_val - 2 - val;
    }
    return val;
}
vector<double> readRawImage(const string& filename) {
    vector<unsigned char> buffer(WIDTH * HEIGHT);
    ifstream file(filename, ios::binary);
    file.read(reinterpret_cast<char*>(buffer.data()), WIDTH * HEIGHT);
    file.close();
    vector<double> img(WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        img[i] = static_cast<double>(buffer[i]);
    }
    return img;
}
void writeRawImage(const string& filename, const vector<unsigned char>& img) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(img.data()), WIDTH * HEIGHT);
    file.close();
}

vector<double> convolve2D(const vector<double>& img, const vector<double>& filter) {
    vector<double> result(WIDTH * HEIGHT, 0.0);
    int offset=2;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double sum=0.0;
            for (int fy = -offset; fy <= offset; ++fy) {
                for (int fx = -offset; fx <= offset; ++fx) {
                    int ny = mirrorBoundary(y + fy, HEIGHT);
                    int nx = mirrorBoundary(x + fx, WIDTH);
                    double pixelVal = img[ny * WIDTH + nx];
                    double filterVal = filter[(fy + offset) * 5 + (fx + offset)];
                    sum += pixelVal * filterVal;
                }
            }
            result[y * WIDTH + x] = sum;
        }
    }
    return result;
}

vector<double> computeEnergy(const vector<double>& response) {
    vector<double> energy(WIDTH * HEIGHT, 0.0);
    int offset = WINDOW_SIZE / 2;
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
            energy[y*WIDTH + x] = sum/(WINDOW_SIZE * WINDOW_SIZE);
        }
    }
    return energy;
}

vector<int> kmeans(const vector<vector<double>>& features, int num_clusters, int max_iters = 50) {
    int num_pixels = WIDTH * HEIGHT;
    int num_features = features[0].size();
    srand(static_cast<unsigned>(time(0)));
    vector<vector<double>> centroids(num_clusters, vector<double>(num_features));
    for (int k = 0; k < num_clusters; ++k) {
        int rand_idx = rand() % num_pixels;
        for (int f = 0; f < num_features; ++f) {
            centroids[k][f] = features[rand_idx][f];
        }
    }
    vector<int> labels(num_pixels, 0);
    bool changed = true;
    int iter = 0;
    while (changed && iter < max_iters) {
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
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                changed = true;
            }
        }
        vector<vector<double>> new_centroids(num_clusters, vector<double>(num_features, 0.0));
        vector<int> counts(num_clusters, 0);
        for (int i = 0; i < num_pixels; ++i) {
            int cluster = labels[i];
            counts[cluster]++;
            for (int f = 0; f < num_features; ++f) {
                new_centroids[cluster][f] += features[i][f];
            }
        }
        for (int k = 0; k < num_clusters; ++k) {
            if (counts[k] > 0) {
                for (int f = 0; f < num_features; ++f) {
                    centroids[k][f] = new_centroids[k][f] / counts[k];
                }
            }
        }
        iter++;
    }
    return labels;
}

int main() {
    vector<double> img = readRawImage("Mosaic.raw");
    vector<vector<double>> filters(NUM_FILTERS, vector<double>(25));
    int filter_idx = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) {
                    filters[filter_idx][y * 5 + x] = laws1D[i][y] * laws1D[j][x];
                }
            }
            filter_idx++;
        }
    }
    vector<vector<double>> energies(NUM_FILTERS);
    for (int i = 0; i < NUM_FILTERS; ++i) {
        vector<double> response = convolve2D(img, filters[i]);
        energies[i] = computeEnergy(response);
    }
    vector<vector<double>> features(WIDTH * HEIGHT, vector<double>(24));
    for (int p = 0; p < WIDTH * HEIGHT; ++p) {
        double L5L5_energy = energies[0][p];
        if (L5L5_energy == 0){
            L5L5_energy= 1e-5;
        }
        int feat_idx = 0;
        for (int i = 1; i < NUM_FILTERS; ++i) {
            features[p][feat_idx] = energies[i][p] / L5L5_energy;
            feat_idx++;
        }
    }
    vector<int> labels = kmeans(features, K);
    vector<unsigned char> output_img(WIDTH * HEIGHT);
    int step = 255/(K - 1);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        output_img[i] = static_cast<unsigned char>(labels[i] * step);
    }
    writeRawImage("p2a_output.raw", output_img);
    return 0;
}