
# Image_Processing_Algorithms

## Description
Image_Processing_Algorithms is a desktop application built with PyQt that implements various image processing algorithms from scratch. It allows users to apply operations like contrast stretching, smoothing, edge detection (Canny, Sobel), sharpening, and more.

## Features
- **Segmentation**: Manual and automatic options
- **Contrast Stretching**: Histogram equalization
- **Smoothing**: Linear (Box/Mean Filter) and Non-linear (Min Filter)
- **Edge Detection**: Canny and Sobel filters
- **Sharpening**: Laplacian and enhancement techniques
- **Time Transmission**: Image size, baud rate, grayscale channel calculations
- **Histogram Analysis**: Displays original and output histograms
  
## GUI overview
### Layout
![alt text](https://github.com/AbdelrahmanSamir12/Image-Processing-Algorithms/blob/main/GUI/layout.png "GUI Layout")

### Contract stretching
![alt text](https://github.com/AbdelrahmanSamir12/Image-Processing-Algorithms/blob/main/GUI/contrast_stretching.png "Contract stretching example")

### Canny Edge Detection 
![alt text](https://github.com/AbdelrahmanSamir12/Image-Processing-Algorithms/blob/main/GUI/canny.png "Canny Edge Detection example")

### Sobel Edge Detection
![alt text](https://github.com/AbdelrahmanSamir12/Image-Processing-Algorithms/blob/main/GUI/sobel.png "Sobel Edge Detection example")
## Installation
1. Clone the repository:
   ```sh
   git clone [this repo link]
   cd Image_Processing_Algorithms
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python Main.py
   ```
## Usage
- Open the application and load an image.
- Apply various image processing algorithms using the provided controls.
- Save the processed image if needed.



