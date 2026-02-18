# Deep Learning Applications in Computer Vision

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW3-green)

This project explores the capabilities and limitations of state-of-the-art pre-trained models for fundamental Computer Vision tasks: **Classification**, **Object Detection**, and **Semantic Segmentation**.

## Project Goals
1.  **Pipeline Engineering:** Implementing a comprehensive 12-step preprocessing pipeline.
2.  **Model Inference:** Utilizing standard architectures (**EfficientNet, Faster R-CNN, DeepLabV3+**).
3.  **Critical Analysis:** Evaluating model robustness against noise, rotation, and complex backgrounds.

## Task 1: The Preprocessing Pipeline

We implemented a complex pipeline to transform raw images into model-ready tensors.

> **Note on Augmentation:**
> While heavy augmentations (Random Rotation, Color Jitter) are typically reserved for the *training phase* to improve generalization, we applied them here during *inference* as per the assignment requirements. This serves to demonstrate data manipulation techniques and test the model's robustness against distorted inputs.

**The 12 Steps:**
`NumPy Conversion` $\to$ `Gaussian Blur` $\to$ `Unsharp Masking` $\to$ `CLAHE` $\to$ `Padding` $\to$ `Resize` $\to$ `Random Rotation/Crop` $\to$ `Color Jitter` $\to$ `Tensor Conversion` $\to$ `Normalization` $\to$ `Batching`.

**Visualizing the Pipeline:**
The figure below demonstrates the transformation process.
*(Visual Note: Steps 10-12 involve structural changes to data types and dimensions [Tensor/Batch], so the visual appearance of the image remains unchanged in the plot).*

![Pipeline](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/03-Deep_Learning_Applications/assets/showcase/pipeline.png)

## Models & Analysis

We evaluated the following architectures using `torchvision` and `timm`:

| Task | Model Architecture | Weights | Key Analysis (Observations) |
| :--- | :--- | :--- | :--- |
| **Classification** | **EfficientNet-B4** | ImageNet | Robust to texture changes (e.g., fur) but sensitive to geometric transformations. It confused a rotated *Minivan* with a *Trailer Truck* due to perspective distortion caused by our augmentation pipeline. |
| **Object Detection** | **Faster R-CNN** (ResNet50-FPN) | COCO | Excellent detection of small objects via FPN. However, it produced False Positives on textured surfaces (e.g., classifying a wooden floor as a "Bench"). |
| **Segmentation** | **DeepLabV3+** (ResNet50) | VOC | Superior boundary detection for complex shapes like fur. It struggled with semantic context in some cases, misclassifying a boat's motor as a "Person". |

## Sample Results

Below is a sample output from the Semantic Segmentation task (DeepLabV3+), demonstrating the model's ability to isolate the subject from the background.

![Segmentation](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/03-Deep_Learning_Applications/assets/showcase/segmentation_sample.png)

## File Structure
* `DL_Applications.ipynb`: The main notebook containing the pipeline implementation and inference logic.
* `assets/inputs/`: The raw sample images used for testing.

## How to Run
1.  Install requirements: `pip install torch torchvision timm opencv-python`.
2.  Place your test images in `assets/inputs/`.
3.  Run the notebook to generate visualizations.
