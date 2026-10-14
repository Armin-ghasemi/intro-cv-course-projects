<div align="center">
    <table style="border: none; border-collapse: collapse;">
        <tr>
            <td align="center" width="20%" style="border: none;">
                <img src="https://i.ibb.co/yXKQmtZ/logo1.png" width="120" alt="University of Tehran Logo"/>
            </td>
            <td align="center" width="60%" style="border: none;">
                <h1>Introduction to Computer Vision<br><small>Course Projects & Assignments</small></h1>
            </td>
            <td align="center" width="20%" style="border: none;">
                <img src="https://i.ibb.co/wLjqFkw/logo2.png" width="150" alt="ECE Faculty Logo"/>
            </td>
        </tr>
    </table>
</div>

---

<div align="center">

![Language](https://img.shields.io/badge/Language-Python%203.x-yellow)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Semester](https://img.shields.io/badge/Semester-Fall%202025-orange)
![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)

</div>

## Overview

This repository contains a comprehensive collection of assignments and the final project implemented for the **Introduction to Computer Vision** course. The objective of this repository is to bridge the gap between classical image processing techniques and modern Deep Learning architectures in Computer Vision.

The projects progress systematically from foundational pixel-level manipulations and classical feature extraction using **OpenCV** to building, optimizing, and evaluating Convolutional Neural Networks (CNNs) and Conditional Generative Adversarial Networks (cGANs) using **PyTorch**.

## Academic Context

- **Course:** Introduction to Computer Vision
- **Institution:** University of Tehran, School of Electrical & Computer Engineering (ECE)
- **Semester:** Fall 2025 (1404-1405)
- **Level:** Undergraduate (B.Sc.)

## Repository Structure

This repository is organized into 5 standalone project directories, showcasing a step-by-step progression from classical computer vision to deep generative modeling:

- **[01-Basic_Image_Processing](./01-Basic_Image_Processing):**  
  Foundational digital image processing operations using OpenCV and NumPy, including color space transformations, histogram equalization, spatial filtering, morphological operations, and edge detection.
- **[02-Classical_Captcha_Solver](./02-Classical_Captcha_Solver):**  
  An automated pipeline for breaking and recognizing CAPTCHA images using purely classical computer vision techniques (denoising, thresholding, contour extraction, character segmentation, and template matching).
- **[03-Deep_Learning_Applications](./03-Deep_Learning_Applications):**  
  Exploring modern Deep Learning paradigms and pre-trained vision architectures for practical computer vision tasks and feature representation.
- **[04-Fashion_MNIST](./04-Fashion_MNIST):**  
  A comparative study between Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) on the Fashion-MNIST dataset, demonstrating the role of spatial hierarchy, Batch Normalization, Dropout, and error analysis via confusion matrices.
- **[05-Final_Project_Pix2Pix](./05-Final_Project_Pix2Pix):**  
  End-to-end implementation of a Conditional GAN (**Pix2Pix**: U-Net Generator + PatchGAN Discriminator) for paired image-to-image translation on the **Edges2Shoes** dataset (~50k images), featuring in-memory binary caching, `torch.compile`, Mixed Precision (AMP), and quantitative evaluation (PSNR/SSIM).

## Tech Stack

The following libraries and tools were utilized across the projects:

| Category | Libraries & Tools |
| :--- | :--- |
| **Deep Learning Core** | `PyTorch`, `Torchvision`, `Torch.nn`, `Torch.amp` |
| **Classical CV & Image Processing** | `OpenCV (cv2)`, `Pillow (PIL)`, `Scikit-Image` |
| **Model Inspection & Metrics** | `Torchinfo`, `Torchview`, `Kornia` (SSIM/PSNR), `Scikit-Learn` |
| **Data Processing & High-Speed I/O** | `NumPy`, `Pandas`, `aria2` |
| **Visualization** | `Matplotlib`, `Seaborn`, `Graphviz` |

## Getting Started

To access the code and notebooks, clone the repository using the following command:

```bash
git clone [https://github.com/Armin-ghasemi/intro-cv-course-projects.git](https://github.com/Armin-ghasemi/intro-cv-course-projects.git)
cd intro-cv-course-projects
