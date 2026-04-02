#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>
#include <map>

using namespace std;

const int WIDTH = 512;
const int HEIGHT = 512;
const int NUM_FILTERS = 25;
const int K = 6;              
const int WINDOW_SIZE = 15;   
const int PCA_DIMS = 5;       // Number of principal components to keep

// 1D Laws Vectors
const double laws1D[5][5] = {
    { 1,  4,  6,  4,  1}, // L5 
    {-1, -2,  0,  2,  1}, // E5 
    {-1,  0,  2,  0, -1}, // S5 
    {-1,  2,  0, -2,  1}, // W5 
    { 1, -4,  6, -4,  1}  // R5 
};

// --- Basic Helpers ---
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

// --- Filtering & Energy (Step 1 & 2) ---
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

// --- Advanced Task 1: PCA Implementation ---
// Basic Jacobi eigenvalue algorithm for symmetric matrices
void jacobiEigen(vector<vector<double>>& cov, vector<vector<double>>& evecs, vector<double>& evals, int n) {
    evecs.assign(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) evecs[i][i] = 1.0;
    
    int max_iters = 100;
    for (int iter = 0; iter < max_iters; iter++) {
        double max_val = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (abs(cov[i][j]) > max_val) { max_val = abs(cov[i][j]); p = i; q = j; }
            }
        }
        if (max_val < 1e-9) break;

        double theta = 0.5 * atan2(2.0 * cov[p][q], cov[q][q] - cov[p][p]);
        double c = cos(theta), s = sin(theta);

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double a_ip = cov[i][p], a_iq = cov[i][q];
                cov[i][p] = cov[p][i] = c * a_ip - s * a_iq;
                cov[i][q] = cov[q][i] = s * a_ip + c * a_iq;
            }
            double e_ip = evecs[i][p], e_iq = evecs[i][q];
            evecs[i][p] = c * e_ip - s * e_iq;
            evecs[i][q] = s * e_ip + c * e_iq;
        }
        double a_pp = cov[p][p], a_qq = cov[q][q], a_pq = cov[p][q];
        cov[p][p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
        cov[q][q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
        cov[p][q] = cov[q][p] = 0.0;
    }
    evals.resize(n);
    for (int i = 0; i < n; i++) evals[i] = cov[i][i];
}

vector<vector<double>> applyPCA(vector<vector<double>>& features, int target_dims) {
    int N = features.size();
    int D = features[0].size();
    
    // 1. Center the data
    vector<double> means(D, 0.0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) means[j] += features[i][j];
    for (int j = 0; j < D; j++) means[j] /= N;
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) features[i][j] -= means[j];

    // 2. Covariance Matrix
    vector<vector<double>> cov(D, vector<double>(D, 0.0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = j; k < D; k++) {
                cov[j][k] += (features[i][j] * features[i][k]) / N;
                cov[k][j] = cov[j][k]; // Symmetric
            }
        }
    }

    // 3. Eigen Decomposition
    vector<vector<double>> evecs;
    vector<double> evals;
    jacobiEigen(cov, evecs, evals, D);

    // 4. Sort indices by eigenvalues descending
    vector<pair<double, int>> eigen_pairs;
    for (int i = 0; i < D; i++) eigen_pairs.push_back({evals[i], i});
    sort(eigen_pairs.rbegin(), eigen_pairs.rend());

    // 5. Project onto target dimensions
    vector<vector<double>> reduced(N, vector<double>(target_dims, 0.0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < target_dims; j++) {
            int e_idx = eigen_pairs[j].second;
            for (int k = 0; k < D; k++) {
                reduced[i][j] += features[i][k] * evecs[k][e_idx];
            }
        }
    }
    return reduced;
}

// --- K-Means ---
vector<int> kmeans(const vector<vector<double>>& features, int num_clusters) {
    int num_pixels = features.size();
    int num_features = features[0].size();
    srand(42); 
    
    vector<vector<double>> centroids(num_clusters, vector<double>(num_features));
    for (int k = 0; k < num_clusters; ++k) {
        int rand_idx = rand() % num_pixels;
        centroids[k] = features[rand_idx];
    }

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
        cout << "K-Means Iteration: " << ++iter << endl;
    }
    return labels;
}

// --- Advanced Task 2: Merge Small Holes (Majority Filter) ---
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

// --- Main ---
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
    cout << "Computing responses & energy..." << endl;
    vector<vector<double>> energies(NUM_FILTERS);
    for (int i = 0; i < NUM_FILTERS; ++i) {
        vector<double> response = convolve2D(img, filters[i]);
        energies[i] = computeEnergy(response, WINDOW_SIZE);
    }

    // 3. Normalization (Discard L5L5)
    cout << "Normalizing features..." << endl;
    vector<vector<double>> features(WIDTH * HEIGHT, vector<double>(24));
    for (int p = 0; p < WIDTH * HEIGHT; ++p) {
        double L5L5 = max(energies[0][p], 1e-5); 
        for (int i = 1; i < NUM_FILTERS; ++i) features[p][i - 1] = energies[i][p] / L5L5;
    }

    // 4. Advanced: PCA Feature Reduction
    cout << "Running PCA to reduce 24D to " << PCA_DIMS << "D..." << endl;
    vector<vector<double>> reduced_features = applyPCA(features, PCA_DIMS);

    // 5. Segmentation
    cout << "Running K-Means on PCA features..." << endl;
    vector<int> labels = kmeans(reduced_features, K);

    // 6. Advanced: Merge Small Holes & Enhance Boundaries
    cout << "Applying post-processing (Majority Filter)..." << endl;
    // Applying a 9x9 Mode Filter significantly closes small noisy holes 
    // and inherently smooths/enhances the boundaries between textures.
    labels = majorityFilter(labels, 9); 

    // 7. Output
    vector<unsigned char> output_img(WIDTH * HEIGHT);
    int step = 255 / (K - 1);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        output_img[i] = static_cast<unsigned char>(labels[i] * step);
    }

    cout << "Writing Advanced_Output.raw..." << endl;
    writeRawImage("Advanced_Output.raw", output_img);
    cout << "Done!" << endl;

    return 0;
}
