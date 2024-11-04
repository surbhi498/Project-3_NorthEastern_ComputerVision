# Project-3_NorthEastern_ComputerVision

# Image Processing and Classification Project

## Overview

This project involves the classification of feature vectors extracted from images using different distance metrics. The primary goal is to analyze and classify regions in images based on their features, such as area, aspect ratio, percent filled, and the axis of least central moment. The project includes several key components such as loading a known objects database, computing distances, classifying feature vectors, and evaluating classification performance using confusion matrices.

## Features

1. **Image Processing Techniques:**
   - **Preprocessing:** Convert images to grayscale and apply Gaussian blur to reduce noise and smooth the image.
   - **Thresholding:** Use adaptive thresholding to create binary images based on local mean values.
   - **Morphological Operations:** Apply closing and opening operations to clean binary images by removing noise and small artifacts.

2. **Feature Extraction:**
   - **Feature Selection:** Compute features such as area, aspect ratio, percent filled, and the axis of least central moment.
   - **Feature Normalization:** Normalize features to ensure equal contribution to distance calculations, particularly for scaled Euclidean distance.

3. **Distance Metrics:**
   - **Simple Euclidean Distance:** Calculate the straight-line distance between two points in feature space.
   - **Scaled Euclidean Distance:** Normalize each feature by its standard deviation before computing the distance.

4. **Classification Techniques:**
   - **Nearest Neighbor Classification:** Assign the label of the closest known feature vector to a new feature vector.
   - **Confusion Matrix:** Use confusion matrices to evaluate classification performance, showing both correct classifications and misclassifications.

5. **Evaluation Metrics:**
   - **Confusion Matrix:** Provide a comprehensive view of classification performance.
   - **Performance Comparison:** Evaluate the performance of different classifiers using confusion matrices to identify the most effective classification approach.

6. **Modular Code Design:**
   - **Function Separation:** Separate functionality into distinct functions for loading data, computing distances, classifying feature vectors, and computing confusion matrices.
   - **Error Handling:** Include error handling for file operations and input validation to improve robustness and user experience.

7. **Data Handling:**
   - **Loading Data:** Efficiently load and parse data from CSV files to create feature vectors for classification.
   - **Standard Deviation Calculation:** Compute the standard deviations of features in the known objects database for normalization.

8. **Visualization:**
   - **Region Visualization:** Visualize regions with bounding boxes, centroids, and feature information to understand the properties of each region.
   - **Classification Results:** Visualize the classification results on the images for an intuitive understanding of the classifier's performance.

## Requirements

- OpenCV
- C++ Standard Library

## Installation

1. **Install OpenCV:**
   - On macOS:
     ```sh
     brew install opencv
     ```
   - On Ubuntu:
     ```sh
     sudo apt-get install libopencv-dev
     ```

2. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>

# Compilation and Execution
Navigate to the Project Directory:
cd /path/to/your/project/directory
# Compile the Code:
```sh
g++ -o task6 task6.cpp `pkg-config --cflags --libs opencv4`

#  Run the Compiled Binary
```sh
./task6 <input_directory> <output_directory> <min_region_size> <max_regions> <feature_file>
<input_directory>: The directory containing the input images.
<output_directory>: The directory where the output images will be saved.
<min_region_size>: The minimum size of regions to be considered.
<max_regions>: The maximum number of regions to process.
<feature_file>: The path to the feature file containing known objects.

Acknowledgements
I would like to acknowledge the following for their contributions and support throughout this project:

OpenCV Library: The OpenCV library provided essential functions for image processing, including preprocessing, thresholding, and morphological operations.
Standard C++ Libraries: The standard C++ libraries facilitated file operations, mathematical calculations, and data handling.
Mentors and Instructors: Special thanks to my mentors and instructors for their guidance and support in understanding the concepts of image processing and classification.
Peers and Collaborators: I appreciate the feedback and collaboration from my peers, which helped in refining the project and improving its overall quality.

Summary
This project provided a comprehensive learning experience in image processing, feature extraction, classification techniques, and evaluation metrics. It emphasized the importance of preprocessing, feature selection, and normalization in achieving effective classification. The project also highlighted the value of modular code design and error handling in creating robust and maintainable code. Overall, the project reinforced key concepts in computer vision and machine learning, providing a solid foundation for further exploration and development in these fields.
