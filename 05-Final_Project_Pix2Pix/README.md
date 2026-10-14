# Image-to-Image Translation with Conditional GAN (Pix2Pix: Edges2Shoes)

![University](https://img.shields.io/badge/University-University%20of%20Tehran-red)
![Course](https://img.shields.io/badge/Course-Introduction%20to%20Computer%20Vision-blue)
![Assignment](https://img.shields.io/badge/Assignment-Final%20Project-green)

An end-to-end Deep Generative Modeling project using **PyTorch** to perform paired image-to-image translation on the **Edges2Shoes** dataset. Inspired by the seminal **Pix2Pix** framework (*Isola et al.*), this project implements a Conditional Generative Adversarial Network (cGAN) pairing a **U-Net Generator** with a **PatchGAN Discriminator** to synthesize realistic shoe photographs directly from binary edge sketches.

## Project Overview

The objective of this final project is to design, optimize, and evaluate a conditional generative pipeline capable of translating structural hand-drawn/extracted edge maps into photorealistic RGB images:

1. **The Dataset (Edges2Shoes):** Consists of **49,825 training** and **200 validation** paired RGB images (`512x256` resolution), where the left half (`256x256`) represents the input edge contour and the right half (`256x256`) contains the ground-truth photograph.
2. **The Generator (U-Net):** An encoder-decoder architecture equipped with **Skip Connections** that shuttle low-level spatial information directly across the bottleneck, ensuring precise structural alignment between the input sketch and generated shoe.
3. **The Discriminator (PatchGAN):** A convolutional classifier that evaluates $N \times N$ local image patches (`32x32` output validity grid) rather than the whole image at once, enforcing sharp high-frequency textures and realistic material details.

Both networks are trained adversarially using the **Adam** optimizer ($\beta_1 = 0.5, \beta_2 = 0.999$) under a combined **cGAN + L1** objective function:

$$\mathcal{L}_{G} = \mathcal{L}_{GAN}(G, D) + \lambda \mathcal{L}_{L1}(G) \quad (\text{where } \lambda = 100)$$

---

## Methodology & Engineering Pipeline

Training a GAN on nearly 50,000 high-resolution images within single-GPU cloud constraints (Google Colab) requires careful systems-level optimization alongside architectural tuning.

### Step 1: High-Performance Data Pipeline (`FastBinaryDataset`)
Standard disk I/O becomes a severe bottleneck when reading ~50,000 image files per epoch. To eliminate disk latency:
* **Automated Acquisition:** Utilized `aria2c` with 16 parallel connections for rapid dataset retrieval.
* **In-Memory Binary Caching:** Implemented `FastBinaryDataset` to preload all compressed JPEG byte streams directly into system RAM at initialization, decoding and splitting pairs on-the-fly via OpenCV (`cv2.imdecode`) and normalizing tensors to $[-1, 1]$.

![Dataset Samples](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/05-Final_Project_Pix2Pix/assets/showcase/dataset_samples.png)

### Step 2: Network Architectures (U-Net vs. PatchGAN)
We designed a memory-efficient variant of the Pix2Pix architecture to balance representational capacity with training throughput, verified using `torchinfo` and `torchview`:

| Component | Architecture Type | Input Tensor Shape | Output Tensor Shape | Trainable Params | Architecture Graph (PDF) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generator ($G$)** | 8-Layer Encoder-Decoder **U-Net** | `(B, 3, 256, 256)` | `(B, 3, 256, 256)` | **14,891,267** (~14.9M) | [📄 View Generator Graph](https://github.com/Armin-ghasemi/intro-cv-course-projects/blob/main/05-Final_Project_Pix2Pix/assets/architectures/generator_architecture.pdf) |
| **Discriminator ($D$)** | Conditional **PatchGAN** | `(B, 6, 256, 256)` | `(B, 1, 32, 32)` | **432,097** (~432K) | [📄 View Discriminator Graph](https://github.com/Armin-ghasemi/intro-cv-course-projects/blob/main/05-Final_Project_Pix2Pix/assets/architectures/discriminator_architecture.pdf) |

* **Generator Details:** 8 downsampling blocks (`Conv2d` + `BatchNorm2d` + `LeakyReLU(0.2)`) compressing spatial dimensions down to a `1x1x256` bottleneck, followed by 7 upsampling blocks (`ConvTranspose2d` + `BatchNorm2d` + `ReLU` + `Dropout(0.5)` on early layers) and a final `Tanh` activation.
* **Discriminator Details:** Concatenates the 3-channel input edge map and 3-channel shoe image into a 6-channel tensor, passing it through strided convolutions to output a `32x32` patch validity matrix.
* *(Note: High-resolution vector diagrams of both network computational graphs generated via `torchview` are attached in the `assets/architectures/` directory).*

### Step 3: Training Acceleration & Dynamics
To train for **50 epochs** (Batch Size = 64) at maximum speed, we leveraged **PyTorch 2.0 `torch.compile`**, **Automatic Mixed Precision (`AMP` with `GradScaler`)**, and `channels_last` memory formatting. A `LambdaLR` scheduler kept the learning rate constant at `0.0002` for the first 25 epochs before linearly decaying it to zero over the final 25 epochs.

**Observation on Loss Curves:**
* **Generator Loss (Left):** Steadily decreases from **15.93** (Epoch 0) to **~8.12** (Epoch 49), demonstrating consistent L1 structural convergence, with noticeable stabilization after learning rate decay begins at Epoch 25.
* **Discriminator Loss (Right):** Oscillates tightly around the theoretical Nash Equilibrium of **0.5** (ending at **0.54**), confirming balanced adversarial training without mode collapse or discriminator dominance.

![Training Loss Curves](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/05-Final_Project_Pix2Pix/assets/showcase/loss_curves.png)

---

## Evaluation & Results

We evaluated the trained checkpoints saved every 5 epochs across the validation set using both **Qualitative Visual Inspection** and **Quantitative Metrics** (`MSE`, `MAE`, `PSNR`, and `SSIM` via `kornia`).

### 1. Qualitative Evolution Across Epochs
Tracking fixed validation sketches from Epoch 0 to Epoch 49 illustrates the progressive refinement of the generator:
* **Epoch 0:** Coarse, blurry color blobs with noticeable grid artifacts.
* **Epochs 5–20:** Rapid acquisition of global shoe geometry, sole boundaries, and primary color fills.
* **Epochs 25–49 (LR Decay Phase):** Fine-tuning of specular highlights, leather/fabric shading, shoelaces, and sharp edge boundaries.

![Training Progression Sample](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/05-Final_Project_Pix2Pix/assets/showcase/training_progression.png)

**Full 10-Sample Evolution Matrix (Epoch 0 to Final Epoch 49):**

![Qualitative Evolution](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/05-Final_Project_Pix2Pix/assets/showcase/qualitative_evolution.png)

### 2. Quantitative Metrics & Progression
Every checkpoint was benchmarked on the 200 validation pairs:

| Epoch | MSE $\downarrow$ | MAE (L1) $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch 0** | 1.4285 | 0.8990 | 4.47 | 0.2724 |
| **Epoch 5** | 0.2073 | 0.2172 | 12.87 | 0.7407 |
| **Epoch 10** | 0.1908 | 0.2089 | 13.22 | 0.7331 |
| **Epoch 15** | 0.1511 | 0.1893 | 14.35 | 0.6961 |
| **Epoch 20** | 0.0792 | 0.1229 | 17.23 | 0.7833 |
| **Epoch 25** | 0.1692 | 0.2098 | 13.75 | 0.6131 |
| **Epoch 30** | 0.0525 | 0.1006 | 18.95 | 0.7815 |
| **Epoch 35 (Best SSIM)** | **0.0496** | 0.1005 | 19.19 | **0.7858** |
| **Epoch 40** | 0.0704 | 0.1153 | 17.73 | 0.7634 |
| **Epoch 45 (Best MAE/PSNR)** | 0.0497 | **0.0984** | **19.21** | 0.7827 |
| **Epoch 49 (Final)** | 0.0500 | 0.0991 | 19.19 | 0.7800 |

![Quantitative Metrics Curves](https://raw.githubusercontent.com/Armin-ghasemi/intro-cv-course-projects/main/05-Final_Project_Pix2Pix/assets/showcase/quantitative_metrics.png)

### 3. Discussion: The PSNR vs. Perceptual Sharpness Trade-off
* **Strong Structural Fidelity (SSIM = 0.7858):** The model comfortably surpasses the target threshold of `0.60`, proving that U-Net's skip connections preserve structural layout, luminance, and contrast with high precision.
* **Understanding the PSNR Score (~19.21 dB):** While pixel-level PSNR stabilizes around 19.2 dB, this behavior is well-documented in generative literature (*Isola et al., 2017*). Because PSNR is directly inversely proportional to **MSE**, optimizing for high PSNR mathematically rewards predicting the *smooth, blurry average* of all plausible colorings. Conversely, our **PatchGAN** discriminator forces the generator to commit to sharp, high-frequency textures (e.g., stitching, buckles, reflections). Even when a realistic generated shoe differs from the ground truth by a slight color shade or texture shift, PSNR penalizes it heavily despite high visual realism.

---

## File Structure

- `edges2shoes.ipynb`: The complete Jupyter Notebook containing high-speed data acquisition, RAM-cached dataset implementation, U-Net & PatchGAN architectures, AMP/compiled training loop, and quantitative/qualitative evaluations.
- `assets/showcase/`: Directory containing dataset samples, training loss curves, progression grids, and evaluation metric plots.
- `assets/architectures/`: Directory containing detailed vector PDF diagrams (`generator_architecture.pdf` and `discriminator_architecture.pdf`) generated via `torchview`.

## How to Run
1. Ensure you have `torch >= 2.0`, `torchvision`, `torchinfo`, `torchview`, `kornia`, `opencv-python`, `pandas`, and `matplotlib` installed.
2. Run `edges2shoes.ipynb` in a CUDA-enabled environment (e.g., Google Colab T4/L4 GPU with sufficient system RAM for dataset caching).
3. The notebook automatically installs system dependencies (`aria2`, `graphviz`), downloads the Edges2Shoes dataset, compiles the networks, and logs checkpoints every 5 epochs.
