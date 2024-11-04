#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <random>
#include <numeric>
#include <map>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>

namespace fs = std::filesystem;

// Structure to store region information
struct Region {
    cv::Point2d centroid;
    mutable cv::Vec3b color;  // Mutable since it's just for visualization
    int area;
    cv::Rect boundingBox;
    double aspectRatio;
    bool touchesBoundary;
    double percentFilled;
    double leastCentralMomentAxis;
};

// Structure to store feature vector and label
struct FeatureVector {
    std::string label;
    int area;
    double aspectRatio;
    double percentFilled;
    double leastCentralMomentAxis;
};

// Function to load the known objects database from a CSV file
std::vector<FeatureVector> load_known_objects(const std::string& filename) {
    std::vector<FeatureVector> known_objects;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return known_objects;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        FeatureVector fv;
        std::getline(ss, fv.label, ',');
        ss >> fv.area;
        ss.ignore(1);
        ss >> fv.aspectRatio;
        ss.ignore(1);
        ss >> fv.percentFilled;
        ss.ignore(1);
        ss >> fv.leastCentralMomentAxis;
        known_objects.push_back(fv);
    }

    file.close();
    return known_objects;
}

// Function to compute the scaled Euclidean distance between two feature vectors
double compute_scaled_euclidean_distance(const FeatureVector& fv1, const FeatureVector& fv2, const std::vector<double>& stdevs) {
    double distance = 0.0;
    distance += std::pow((fv1.area - fv2.area) / stdevs[0], 2);
    distance += std::pow((fv1.aspectRatio - fv2.aspectRatio) / stdevs[1], 2);
    distance += std::pow((fv1.percentFilled - fv2.percentFilled) / stdevs[2], 2);
    distance += std::pow((fv1.leastCentralMomentAxis - fv2.leastCentralMomentAxis) / stdevs[3], 2);
    return std::sqrt(distance);
}

// Function to compute the simple Euclidean distance between two feature vectors
double compute_euclidean_distance(const FeatureVector& fv1, const FeatureVector& fv2) {
    double distance = 0.0;
    distance += std::pow(fv1.area - fv2.area, 2);
    distance += std::pow(fv1.aspectRatio - fv2.aspectRatio, 2);
    distance += std::pow(fv1.percentFilled - fv2.percentFilled, 2);
    distance += std::pow(fv1.leastCentralMomentAxis - fv2.leastCentralMomentAxis, 2);
    return std::sqrt(distance);
}

// Function to compute the standard deviations of the features in the known objects database
std::vector<double> compute_feature_stdevs(const std::vector<FeatureVector>& known_objects) {
    std::vector<double> means(4, 0.0);
    std::vector<double> stdevs(4, 0.0);

    for (const auto& fv : known_objects) {
        means[0] += fv.area;
        means[1] += fv.aspectRatio;
        means[2] += fv.percentFilled;
        means[3] += fv.leastCentralMomentAxis;
    }

    for (auto& mean : means) {
        mean /= known_objects.size();
    }

    for (const auto& fv : known_objects) {
        stdevs[0] += std::pow(fv.area - means[0], 2);
        stdevs[1] += std::pow(fv.aspectRatio - means[1], 2);
        stdevs[2] += std::pow(fv.percentFilled - means[2], 2);
        stdevs[3] += std::pow(fv.leastCentralMomentAxis - means[3], 2);
    }

    for (auto& stdev : stdevs) {
        stdev = std::sqrt(stdev / known_objects.size());
    }

    return stdevs;
}

// Function to classify a new feature vector using the known objects database
std::string classify_feature_vector(const FeatureVector& fv, const std::vector<FeatureVector>& known_objects, const std::vector<double>& stdevs, bool use_scaled) {
    double min_distance = std::numeric_limits<double>::max();
    std::string best_label;

    for (const auto& known_fv : known_objects) {
        double distance;
        if (use_scaled) {
            distance = compute_scaled_euclidean_distance(fv, known_fv, stdevs);
        } else {
            distance = compute_euclidean_distance(fv, known_fv);
        }
        if (distance < min_distance) {
            min_distance = distance;
            best_label = known_fv.label;
        }
    }

    return best_label;
}

// Function to compute the confusion matrix
std::map<std::string, std::map<std::string, int>> compute_confusion_matrix(const std::vector<FeatureVector>& test_set, const std::vector<FeatureVector>& known_objects, const std::vector<double>& stdevs, bool use_scaled) {
    std::map<std::string, std::map<std::string, int>> confusion_matrix;
    std::set<std::string> labels;

    // Collect all labels
    for (const auto& fv : known_objects) {
        labels.insert(fv.label);
    }

    // Initialize confusion matrix with all possible labels
    for (const auto& actual_label : labels) {
        for (const auto& predicted_label : labels) {
            confusion_matrix[actual_label][predicted_label] = 0;
        }
    }

    // Populate confusion matrix
    for (const auto& test_fv : test_set) {
        std::string predicted_label = classify_feature_vector(test_fv, known_objects, stdevs, use_scaled);
        confusion_matrix[test_fv.label][predicted_label]++;
    }

    return confusion_matrix;
}

// Function to print the confusion matrix
void print_confusion_matrix(const std::map<std::string, std::map<std::string, int>>& confusion_matrix) {
    std::cout << "Confusion Matrix:" << std::endl;
    for (const auto& actual_pair : confusion_matrix) {
        std::cout << actual_pair.first << ": ";
        for (const auto& predicted_pair : actual_pair.second) {
            std::cout << predicted_pair.first << "=" << predicted_pair.second << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory> <min_region_size> <max_regions> <feature_file>" << std::endl;
        return -1;
    }

    try {
        std::string input_directory = argv[1];
        std::string output_directory = argv[2];
        int min_region_size = std::stoi(argv[3]);
        int max_regions = std::stoi(argv[4]);
        std::string feature_file = argv[5];

        if (!fs::is_directory(input_directory)) {
            std::cerr << "Error: Provided input path is not a directory." << std::endl;
            return -1;
        }

        // Load known objects database
        std::vector<FeatureVector> known_objects = load_known_objects(feature_file);
        if (known_objects.empty()) {
            std::cerr << "Error: No known objects loaded from " << feature_file << std::endl;
            return -1;
        }

        // Compute feature standard deviations
        std::vector<double> stdevs = compute_feature_stdevs(known_objects);

        // Split dataset into training and test sets (for simplicity, using the same known_objects as test_set)
        std::vector<FeatureVector> test_set = known_objects;

        // Compute confusion matrices
        auto confusion_matrix_simple = compute_confusion_matrix(test_set, known_objects, stdevs, false);
        auto confusion_matrix_scaled = compute_confusion_matrix(test_set, known_objects, stdevs, true);

        // Print confusion matrices
        std::cout << "Simple Euclidean Distance:" << std::endl;
        print_confusion_matrix(confusion_matrix_simple);

        std::cout << "Scaled Euclidean Distance:" << std::endl;
        print_confusion_matrix(confusion_matrix_scaled);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}