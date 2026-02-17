# Classical CAPTCHA Solver & Generator

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Assignment](https://img.shields.io/badge/Assignment-HW2-green)

An end-to-end Computer Vision pipeline designed to generate, corrupt, and break text-based CAPTCHAs using **Classical Image Processing** techniques (Morphology, Filtering, and Template Matching), without relying on Deep Learning.

## Project Overview

The goal of this project is to simulate an adversarial scenario:
1.  **The Generator:** Creates synthetic CAPTCHA images.
2.  **The Adversary:** Corrupts images with **Salt & Pepper Noise** and **Gaussian Blur**.
3.  **The Solver:** A pipeline that restores the image quality and recognizes characters using **Normalized Cross-Correlation**.

## Methodology: The Experiment

Our core challenge was removing heavy noise without destroying character edges. We hypothesized two different filtering pipelines and compared their results:

### Step 1: Restoration Experiment
We tested two sequences to restore the corrupted image:

* **Method A (Median $\to$ Sharpen):** First remove noise, then sharpen edges.
* **Method B (Sharpen $\to$ Median):** First enhance edges (and noise), then remove noise.

**Observation:**
Method A failed because the Median filter blurred the already weak edges.
**Method B won** because Sharpening first made the character edges "strong enough" to survive the subsequent Median blur.

| Input (Noisy & Blurred) | Method A: Median $\to$ Sharpen | Method B: Sharpen $\to$ Median (Selected) |
| :---: | :---: | :---: |
| ![Input](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/1_input.png) | ![Method A](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/2a_median_sharpen.png) | ![Method B](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/2b_sharpen_median.png) |
| *Heavy Distortion* | *Failed: Edges Lost* | *Success: Edges Preserved* |

### Step 2: Recognition (Template Matching)

Once the image is restored (Method B) and binarized, we segment individual characters. The final recognition is done by comparing each extracted segment against our `Mapset` templates using correlation.

**Visualizing the Match:**
The table below displays a real example from our dataset (Captcha: `fa0`). The extracted characters (Top) are matched against the Mapset templates (Bottom) with high confidence scores.

| Character 1 | Character 2 | Character 3 |
| :---: | :---: | :---: |
| ![Ex1](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/extracted_1.png) | ![Ex2](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/extracted_2.png) | ![Ex3](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/extracted_3.png) |
| $\downarrow$ *Best Match* $\downarrow$ | $\downarrow$ *Best Match* $\downarrow$ | $\downarrow$ *Best Match* $\downarrow$ |
| ![Ref1](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/template_1.png) | ![Ref2](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/template_2.png) | ![Ref3](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/02-Classical_Captcha_Solver/assets/showcase/template_3.png) |
| **Detected: 'f'**<br>Score: 0.8765 | **Detected: 'a'**<br>Score: 0.8756 | **Detected: '0'**<br>Score: 0.8128 |

## File Structure

* `Captcha_Solver.ipynb`: The main notebook containing the experiment logic.
* `Captcha_Generator.py`: Script to generate the synthetic dataset.
* `assets/Mapset/`: Reference character templates required for matching.

## How to Run
1.  Ensure `assets/` contains `Mapset` and `Captcha` (Input) folders.
2.  Run `Captcha_Solver.ipynb`.
3.  The script will generate an `outputs/` directory containing all intermediate and final results.
