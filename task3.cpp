#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <random>
#include <numeric>
#include <map>
#include <vector>
#include <cmath>

namespace fs = std::filesystem;

cv::Mat preprocess_image(const cv::Mat& frame) {
    cv::Mat gray, blurred;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    return blurred;
}

cv::Mat adaptive_threshold(const cv::Mat& blurred) {
    cv::Mat thresholded;
    cv::adaptiveThreshold(blurred, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY_INV, 11, 2);
    return thresholded;
}

cv::Mat clean_image(const cv::Mat& thresholded) {
    cv::Mat cleaned;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresholded, cleaned, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
    return cleaned;
}

cv::Mat create_region_map(const cv::Mat& cleaned, int min_region_size, int max_regions) {
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids);

    // Create an output image with random colors for each region
    cv::Mat region_map = cv::Mat::zeros(cleaned.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(num_labels);
    std::mt19937 rng(12345); // Random number generator with a fixed seed for reproducibility
    for (int i = 1; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_region_size) {
            colors[i] = cv::Vec3b(rng() % 256, rng() % 256, rng() % 256);
        } else {
            colors[i] = cv::Vec3b(0, 0, 0); // Ignore small regions
        }
    }

    // Sort regions by size and limit to the largest N regions
    std::vector<int> sorted_indices(num_labels - 1);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 1);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&stats](int a, int b) {
        return stats.at<int>(a, cv::CC_STAT_AREA) > stats.at<int>(b, cv::CC_STAT_AREA);
    });

    for (int i = 0; i < std::min(max_regions, static_cast<int>(sorted_indices.size())); ++i) {
        int label = sorted_indices[i];
        int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        int top = stats.at<int>(label, cv::CC_STAT_TOP);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        // Check if the region is central and does not touch the boundary
        if (left > 0 && top > 0 && (left + width) < cleaned.cols && (top + height) < cleaned.rows) {
            for (int y = 0; y < labels.rows; ++y) {
                for (int x = 0; x < labels.cols; ++x) {
                    if (labels.at<int>(y, x) == label) {
                        region_map.at<cv::Vec3b>(y, x) = colors[label];
                    }
                }
            }
        }
    }

    return region_map;
}

void process_images(const std::string& input_directory, const std::string& output_directory, int min_region_size, int max_regions) {
    // Ensure the output directory exists
    if (!fs::exists(output_directory)) {
        fs::create_directory(output_directory);
    }

    for (int i = 1; ; ++i) {
        std::string image_name = "img" + std::to_string(i) + "p3.png";
        std::string input_path = input_directory + "/" + image_name;
        std::string output_path = output_directory + "/" + image_name;

        std::cout << "Processing: " << input_path << std::endl;

        if (!fs::exists(input_path)) {
            std::cout << "Image not found: " << input_path << std::endl;
            break; // Stop if the image does not exist
        }

        cv::Mat frame = cv::imread(input_path);
        if (frame.empty()) {
            std::cerr << "Error: Could not read image file " << input_path << std::endl;
            continue;
        }

        cv::Mat blurred = preprocess_image(frame);
        cv::Mat thresholded = adaptive_threshold(blurred);
        cv::Mat cleaned = clean_image(thresholded);
        cv::Mat region_map = create_region_map(cleaned, min_region_size, max_regions);

        cv::imshow("Thresholded Image", thresholded);
        cv::imshow("Cleaned Image", cleaned);
        cv::imshow("Region Map", region_map);
        cv::waitKey(0); // Wait for a key press to move to the next image

        // Save the region map
        std::cout << "Saving: " << output_path << std::endl;
        if (!cv::imwrite(output_path, region_map)) {
            std::cerr << "Error: Could not save image to " << output_path << std::endl;
        }
    }

    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory> <min_region_size> <max_regions>" << std::endl;
        return -1;
    }

    std::string input_directory = argv[1];
    std::string output_directory = argv[2];
    int min_region_size = std::stoi(argv[3]);
    int max_regions = std::stoi(argv[4]);

    if (fs::is_directory(input_directory)) {
        process_images(input_directory, output_directory, min_region_size, max_regions);
    } else {
        std::cerr << "Error: Provided input path is not a directory." << std::endl;
        return -1;
    }

    return 0;
}