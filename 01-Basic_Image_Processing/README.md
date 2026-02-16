# Basic Image Processing & Filtering

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW1-green)

This folder contains the implementation of fundamental image processing techniques, focusing on pixel-level manipulation, noise reduction, and edge detection using **OpenCV** and **NumPy**.

## Project Overview
In this assignment, we move beyond high-level libraries to understand how digital images are structured and manipulated as matrices. We explore how different filters affect image quality and structure.

**Key Concepts Implemented:**
* **Color Space Analysis:** Converting between RGB, Grayscale, Binary, and HSV spaces.
* **Histogram Processing:** Implementing **Histogram Equalization** and **Stretching** to improve image contrast.
* **Noise Simulation & Filtering:**
    * Adding artificial noise (Salt & Pepper, Gaussian).
    * Comparing **Linear Filters** (Mean, Gaussian) vs. **Non-Linear Filters** (Median).
* **Edge Detection:** Extracting structural features using Sobel, Laplacian, and the Canny algorithm.

> **Note on Filter Performance:**
> A key observation in this project is the performance of the **Median Filter** against Salt & Pepper noise.
> * **Linear Filters:** Tend to "smudge" the noise pixels.
> * **Non-Linear Filter (Median):** Completely eliminates the noise outliers while preserving sharp edges.

## Files
* `Image_Processing_Basics.ipynb`: The main notebook containing the code and visualizations.
* `assets/`: Contains the input images and result comparisons.

## Visualization

### 1. Noise Reduction Comparison
Comparing the original image, noisy version, and the restored version using the Median Filter.

| Original Image | + Salt & Pepper Noise | Denoised (Median Filter) |
| :---: | :---: | :---: |
| ![Original](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/01-Basic_Image_Processing/assets/Pic.jpg) | ![Noisy](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/01-Basic_Image_Processing/assets/showcase_noise_sp.jpg) | ![Denoised](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/01-Basic_Image_Processing/assets/showcase_denoised.jpg) |

### 2. Edge Detection (Canny)
Extracting clean edges from the grayscale image.

| Grayscale Input | Canny Edge Output |
| :---: | :---: |
| ![Gray](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/01-Basic_Image_Processing/assets/showcase_gray.jpg) | ![Canny](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/01-Basic_Image_Processing/assets/showcase_canny.jpg) |

## How to run
1.  Open `Image_Processing_Basics.ipynb`.
2.  Install dependencies: `pip install opencv-python matplotlib numpy`
3.  Run all cells to regenerate outputs in the `outputs/` folder.
