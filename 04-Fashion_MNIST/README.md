# Fashion-MNIST Classification: MLP vs. CNN

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW4-green)

An end-to-end Deep Learning project using **PyTorch** to classify clothing items from the Fashion-MNIST dataset. This project explores and compares two fundamental neural network architectures: a standard Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), highlighting the importance of spatial feature extraction in Computer Vision.

## Project Overview

The goal of this assignment is to understand how different network architectures process image data:
1.  **The Dataset:** Fashion-MNIST, consisting of 28x28 grayscale images across 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
2.  **The Baseline Model (MLP):** Flattens the 2D image into a 1D vector, treating pixels as independent features and losing spatial hierarchy.
3.  **The Improved Model (CNN):** Utilizes convolutional layers to preserve spatial structure and detect visual features like edges, textures, and shapes.

Both models are optimized using the **Adam** optimizer and trained against the Cross-Entropy loss function:
$$Loss = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$

## Methodology: The Experiment

Our core challenge was to improve classification accuracy by transitioning from a basic fully connected network to a deep convolutional architecture. We hypothesized that the CNN would significantly outperform the MLP by avoiding the initial flattening process.

### Step 1: Architecture Comparison
We built and trained both models for 20 epochs. To enhance the CNN, we introduced **Batch Normalization** (for stability), **Dropout** (for regularization), and a **ReduceLROnPlateau Scheduler** (to fine-tune the learning rate).

**Observation:**
The MLP provided a solid baseline, but the **CNN won** with a significantly higher accuracy. Despite having more parameters, the CNN's ability to "see" spatial relationships proved crucial.

| Feature | Baseline: MLP Model | Improved: CNN Model (Selected) |
| :--- | :---: | :---: |
| **Input Handling** | Flattened (1D Vector: 784) | Spatial (2D Matrix: 1x28x28) |
| **Trainable Params** | ~235K | ~1.9M |
| **Test Accuracy** | **88.73%** | **92.67%** |

### Step 2: Qualitative Results & Error Analysis

To truly understand the difference between the models, we must look beyond raw accuracy and analyze *what* they are getting wrong.

**2.1 Confusion Matrices:**
The confusion matrices reveal that visually similar classes (like Shirt vs. T-shirt, or Coat vs. Pullover) are the main source of error. However, the CNN drastically reduces these misclassifications because it can detect structural details (like zippers or collars) that the MLP loses during image flattening.

| Baseline: MLP Confusion Matrix | Improved: CNN Confusion Matrix |
| :---: | :---: |
| ![MLP CM](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/04-Fashion_MNIST/assets/showcase/mlp_cm_crop.png) | ![CNN CM](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/04-Fashion_MNIST/assets/showcase/cnn_cm_crop.png) |

**2.2 Prediction Samples (Correct vs. Incorrect):**
Visualizing the predictions gives us an intuition about the models' behavior. Even when the CNN fails, its mistakes are often more "reasonable" (e.g., confusing two very similar tops) compared to the MLP. The images below display 5 correct (top row) and 5 incorrect (bottom row) predictions for each model.

| Model | Prediction Samples (Correct & Incorrect) |
| :---: | :---: |
| **MLP** | ![MLP Predictions](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/04-Fashion_MNIST/assets/showcase/mlp_predictions.png) |
| **CNN** | ![CNN Predictions](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/04-Fashion_MNIST/assets/showcase/cnn_predictions.png) |

### Step 3: Training Dynamics & Overfitting
While the CNN achieved higher accuracy, analyzing the loss curves revealed that training for 20 epochs caused **overfitting**. The CNN's training loss approached zero, but the validation loss formed a "U-shape", plateauing and then increasing after Epoch 5. This indicates that the optimal model weights were actually achieved early in the training process.

## File Structure

* `Fashion_MNIST.ipynb`: The main notebook containing data loading, model definitions, training loops, and evaluation metrics.
* `assets/`: Folder containing saved plots, prediction samples, and confusion matrix images for documentation.

## How to Run
1.  Ensure you have `torch`, `torchvision`, `torchinfo`, `seaborn`, and `matplotlib` installed.
2.  Run `Fashion_MNIST.ipynb` in a Jupyter environment.
3.  The notebook is configured to automatically detect and use a CUDA-enabled GPU if available, falling back to CPU otherwise.
