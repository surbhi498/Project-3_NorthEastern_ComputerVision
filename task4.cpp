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
    cv::RotatedRect orientedBoundingBox;
};

// Class to track regions across frames
class RegionTracker {
private:
    std::vector<Region> previousRegions;
    std::mt19937 rng;
    const double MAX_CENTROID_DISTANCE = 50.0;

    // Generate a random color
    cv::Vec3b generateRandomColor() {
        return cv::Vec3b(rng() % 256, rng() % 256, rng() % 256);
    }

    // Calculate the Euclidean distance between two points
    double calculateDistance(const cv::Point2d& p1, const cv::Point2d& p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

public:
    RegionTracker() : rng(12345) {}

    // Get the color for a region, matching it with previous regions if possible
    cv::Vec3b getRegionColor(const Region& currentRegion) {
        double minDistance = std::numeric_limits<double>::max();
        cv::Vec3b matchedColor;
        bool found = false;

        for (const auto& prevRegion : previousRegions) {
            double distance = calculateDistance(currentRegion.centroid, prevRegion.centroid);
            if (distance < minDistance && distance < MAX_CENTROID_DISTANCE) {
                minDistance = distance;
                matchedColor = prevRegion.color;
                found = true;
            }
        }

        return found ? matchedColor : generateRandomColor();
    }

    // Update the list of previous regions
    void updateRegions(const std::vector<Region>& newRegions) {
        previousRegions = newRegions;
    }
};

// Preprocess the image by converting it to grayscale and applying Gaussian blur
cv::Mat preprocess_image(const cv::Mat& frame) {
    cv::Mat gray, blurred;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    return blurred;
}

// Apply adaptive thresholding to the blurred image
cv::Mat adaptive_threshold(const cv::Mat& blurred) {
    cv::Mat thresholded;
    cv::adaptiveThreshold(blurred, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY_INV, 11, 2);
    return thresholded;
}

// Clean the thresholded image using morphological operations
cv::Mat clean_image(const cv::Mat& thresholded) {
    cv::Mat cleaned;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresholded, cleaned, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
    return cleaned;
}

// Extract regions from the cleaned image
std::vector<Region> extract_regions(const cv::Mat& cleaned, int min_region_size) {
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids);
    
    std::vector<Region> regions;
    for (int i = 1; i < num_labels; ++i) {
        Region region;
        region.area = stats.at<int>(i, cv::CC_STAT_AREA);
        
        if (region.area < min_region_size) continue;

        region.centroid = cv::Point2d(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        region.boundingBox = cv::Rect(
            stats.at<int>(i, cv::CC_STAT_LEFT),
            stats.at<int>(i, cv::CC_STAT_TOP),
            stats.at<int>(i, cv::CC_STAT_WIDTH),
            stats.at<int>(i, cv::CC_STAT_HEIGHT)
        );
        
        region.aspectRatio = static_cast<double>(region.boundingBox.width) / 
                           static_cast<double>(region.boundingBox.height);
        
        region.touchesBoundary = 
            region.boundingBox.x <= 0 || 
            region.boundingBox.y <= 0 || 
            region.boundingBox.x + region.boundingBox.width >= cleaned.cols ||
            region.boundingBox.y + region.boundingBox.height >= cleaned.rows;

        // Calculate percent filled
        region.percentFilled = static_cast<double>(region.area) / (region.boundingBox.width * region.boundingBox.height);

        // Calculate moments and least central moment axis
        cv::Moments moments = cv::moments(cleaned(region.boundingBox), true);
        double mu20 = moments.mu20 / moments.m00;
        double mu02 = moments.mu02 / moments.m00;
        double mu11 = moments.mu11 / moments.m00;
        region.leastCentralMomentAxis = 0.5 * std::atan2(2 * mu11, mu20 - mu02);

        // Calculate oriented bounding box
        std::vector<cv::Point> points;
        for (int y = region.boundingBox.y; y < region.boundingBox.y + region.boundingBox.height; ++y) {
            for (int x = region.boundingBox.x; x < region.boundingBox.x + region.boundingBox.width; ++x) {
                if (cleaned.at<uchar>(y, x) == 255) {
                    points.emplace_back(x, y);
                }
            }
        }
        region.orientedBoundingBox = cv::minAreaRect(points);

        // Initialize color (will be set properly during visualization)
        region.color = cv::Vec3b(0, 0, 0);

        regions.push_back(region);
    }

    std::sort(regions.begin(), regions.end(), 
              [](const Region& a, const Region& b) { return a.area > b.area; });
              
    return regions;
}

// Draw region information on the output image
void draw_region_information(cv::Mat& output, const Region& region, const cv::Vec3b& color) {
    // Draw bounding box
    cv::rectangle(output, region.boundingBox, color, 2);
    
    // Draw centroid
    cv::circle(output, cv::Point(region.centroid.x, region.centroid.y), 4, color, -1);
    
    // Draw region information
    std::string areaText = "Area: " + std::to_string(region.area);
    std::string aspectText = "AR: " + std::to_string(static_cast<int>(region.aspectRatio * 100) / 100.0);
    std::string percentFilledText = "Filled: " + std::to_string(static_cast<int>(region.percentFilled * 100)) + "%";
    
    cv::putText(output, areaText, 
                cv::Point(region.boundingBox.x, region.boundingBox.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    cv::putText(output, aspectText, 
                cv::Point(region.boundingBox.x, region.boundingBox.y - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    cv::putText(output, percentFilledText, 
                cv::Point(region.boundingBox.x, region.boundingBox.y - 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

    // Draw least central moment axis
    double angle = region.leastCentralMomentAxis;
    double length = std::min(region.boundingBox.width, region.boundingBox.height) / 2.0;
    cv::Point2d start(region.centroid.x - length * std::cos(angle), region.centroid.y - length * std::sin(angle));
    cv::Point2d end(region.centroid.x + length * std::cos(angle), region.centroid.y + length * std::sin(angle));
    cv::line(output, start, end, color, 2);

    // Draw oriented bounding box
    cv::Point2f vertices[4];
    region.orientedBoundingBox.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(output, vertices[i], vertices[(i + 1) % 4], color, 2);
    }
}

// Visualize regions on the original image
cv::Mat visualize_regions(const cv::Mat& original, const cv::Mat& labels, 
                         const std::vector<Region>& regions,
                         RegionTracker& tracker, int max_regions) {
    cv::Mat output = original.clone();
    std::vector<Region> processedRegions;
    
    int processed_count = 0;
    for (const auto& region : regions) {
        if (processed_count >= max_regions || region.touchesBoundary) {
            continue;
        }

        cv::Vec3b color = tracker.getRegionColor(region);
        draw_region_information(output, region, color);
        
        Region processedRegion = region;
        processedRegion.color = color;
        processedRegions.push_back(processedRegion);
        
        processed_count++;
    }
    
    tracker.updateRegions(processedRegions);
    return output;
}

// Save the feature vector of a region to a file
void save_feature_vector(const std::string& filename, const Region& region, const std::string& label) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << label << ","
             << region.area << ","
             << region.aspectRatio << ","
             << region.percentFilled << ","
             << region.leastCentralMomentAxis << "\n";
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}

// Process images in the input directory
void process_images(const std::string& input_directory, const std::string& output_directory, 
                   int min_region_size, int max_regions, const std::string& feature_file) {
    if (!fs::exists(output_directory)) {
        fs::create_directory(output_directory);
    }

    RegionTracker tracker;
    
    for (int i = 1; ; ++i) {
        std::string image_name = "img" + std::to_string(i) + "p3.png";
        std::string input_path = input_directory + "/" + image_name;
        std::string output_path = output_directory + "/" + image_name;

        if (!fs::exists(input_path)) {
            std::cout << "Finished processing all images." << std::endl;
            break;
        }

        std::cout << "Processing: " << input_path << std::endl;

        cv::Mat frame = cv::imread(input_path);
        if (frame.empty()) {
            std::cerr << "Error: Could not read image file " << input_path << std::endl;
            continue;
        }

        // Process image
        cv::Mat blurred = preprocess_image(frame);
        cv::Mat thresholded = adaptive_threshold(blurred);
        cv::Mat cleaned = clean_image(thresholded);
        
        // Extract and visualize regions
        std::vector<Region> regions = extract_regions(cleaned, min_region_size);
        cv::Mat visualization = visualize_regions(frame, cleaned, regions, tracker, max_regions);

        // Display results
        cv::imshow("Original", frame);
        cv::imshow("Processed", visualization);
        cv::imshow("Thresholded", thresholded);
        cv::imshow("Cleaned", cleaned);
        
        char key = cv::waitKey(30);
        if (key == 'n' || key == 'N') {
            std::string label;
            std::cout << "Enter label for the current object: ";
            std::cin >> label;
            for (const auto& region : regions) {
                save_feature_vector(feature_file, region, label);
            }
        } else if (key == 27) { // ESC key
            std::cout << "Processing interrupted by user." << std::endl;
            break;
        }

        // Save output
        if (!cv::imwrite(output_path, visualization)) {
            std::cerr << "Error: Could not save image to " << output_path << std::endl;
        } else {
            std::cout << "Saved: " << output_path << std::endl;
        }
    }

    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory> "
                 << "<min_region_size> <max_regions> <feature_file>" << std::endl;
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

        process_images(input_directory, output_directory, min_region_size, max_regions, feature_file);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}