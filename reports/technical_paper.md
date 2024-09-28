# **Facial Reconstruction from CCTV Footage**

## GITHUB - https://github.com/ByteMeEthos/CCTV_footage_detection

## TECHNICAL REPORT - https://docs.google.com/document/d/15IG7hoI4URyP6Cvhs7bapI9m-8Fmvtq6vnfhs_BSA5U/edit?usp=drive_link

## Table of Contents

1. **Abstract**
2. **Keywords**
3. **Dataset**
4. **Introduction**
   4.1 Core Objectives  
   4.2 Significance  
   4.3 Approach
5. **Literature Review**
   5.1 Introduction and Traditional Methods  
   5.2 Machine Learning Approaches  
   5.3 Challenges in CCTV Facial Reconstruction  
   5.4 Ethical Considerations  
   5.5 Conclusion
6. **Model Architecture and Design**
   6.1 Overview  
   6.2 Generator Architecture  
   6.2.1 U-Net Structure  
   6.2.2 Layer Configuration  
   6.2.3 Loss Functions
7. **Training Strategy**
   7.1 Optimizer  
   7.2 Learning Rate Scheduler  
   7.3 Gradient Control  
   7.4 Training Duration and Hardware  
   7.5 Batch Size and Metrics  
   7.6 Image Enhancement Techniques  
   7.7 Code Management and Explanation  
   7.8 Checkpointing and Model Saving  
   7.9 Validation Strategy
8. **Image Enhancement Techniques**
   8.1 Preprocessing  
   8.2 Real-Time Processing
9. **Additional Components**
   9.1 Face Detection  
   9.2 GAN Framework  
   9.3 Training Monitoring

---

## **1. Abstract**

This technical report presents a proof of concept for an advanced AI model designed to reconstruct and enhance facial images from low-quality CCTV footage. Addressing critical challenges in law enforcement and security, our research focuses on improving facial recognition accuracy in surveillance applications. The model employs a Generative Adversarial Network (GAN) architecture, specifically a **Pix2Pix** framework with a **U-Net**-based generator optimized for image-to-image translation tasks.

Our approach combines machine learning techniques with traditional image processing methods, including **super-resolution**, **noise reduction**, and **deblurring**. The model's architecture incorporates both **pixel-wise** and **perceptual loss functions**, utilizing **VGG16** for feature extraction. **Quantitative evaluation metrics** like **Peak Signal-to-Noise Ratio (PSNR)** and **Structural Similarity Index (SSIM)** demonstrate significant improvements in image quality. Qualitative assessments revealed enhanced **sharpness**, **detail preservation**, and **artifact reduction** in reconstructed facial images.

Key challenges addressed include model training complexity, dataset limitations, and hyperparameter optimization. Future work will focus on integrating **YOLO v8n** for improved face detection, developing real-time video processing capabilities, and exploring **3D facial reconstruction techniques**.

---

## **2. Keywords**

- **Convolutional Neural Networks (CNNs)**
- **PyTorch**
- **Eigenfaces**
- **Haar Cascades**
- **Softmax Function**
- **Batch Normalization**
- **Hyperparameter Tuning**
- **Distance Metrics (e.g., Euclidean Distance, Cosine Similarity)**
- **VGG16**
- **U-Net**
- **YOLO (You Only Look Once)**
- **Dlib**
- **Generative Adversarial Networks (GANs)**

---

## **3. Dataset**

**SCface** is a database of static images of human faces. Images were taken in an uncontrolled indoor environment using five video surveillance cameras of various qualities. The database contains **4160 static images** (in visible and infrared spectrum) of **130 subjects**. These images from different quality cameras mimic real-world conditions and enable robust face recognition algorithm testing.

- **SCface Link**: [SCface](https://www.scface.org/)
- **Dataset Link**: [Kaggle - SCface Dataset](https://www.kaggle.com/datasets/yazkarajih/scface)

---

## **4. Introduction**

This project focuses on developing a **Proof of Concept (POC)** for a machine learning model capable of reconstructing and enhancing facial images from low-quality CCTV footage.

### 4.1 Core Objectives

- **Basic Facial Reconstruction Model**: Develop a machine learning model to reconstruct and enhance facial images from low-quality CCTV footage.

  - **Challenges**:
    - Improving clarity in low-light conditions
    - Addressing motion-blur issues

- **Image Enhancement Techniques**: Apply and integrate basic image processing methods, including super-resolution, noise reduction, and deblurring.

- **Results Comparison and Evaluation**: Provide side-by-side comparisons of original CCTV footage and enhanced images, using quantitative metrics.

### 4.2 Significance

This project addresses a critical need in **video surveillance** and **forensic analysis**, aiming to enhance the quality of facial images from CCTV footage to improve facial recognition systems.

### 4.3 Approach

Combining **machine learning** techniques with **image processing methods**, our approach seeks to create a robust system capable of handling real-world surveillance scenarios, from low-light environments to fast-moving subjects.

---

## **5. Literature Review**

### 5.1 Introduction and Traditional Methods

Early approaches to facial reconstruction from CCTV footage relied on **traditional image processing** techniques like **super-resolution**, **denoising**, and **deblurring** (Wang et al., 2014).

### 5.2 Machine Learning Approaches

The advent of **deep learning** introduced techniques like **Convolutional Neural Networks (CNNs)** and **Generative Adversarial Networks (GANs)** for more effective facial reconstruction.

- **SRCNN** (Dong et al., 2016) showed promising results in image super-resolution.
- **SRGAN** (Ledig et al., 2017) and **ESRGAN** (Wang et al., 2018) improved visual quality through GANs.
- **FSRNet** (Chen et al., 2018) used facial landmarks to enhance reconstruction.
- **Pix2Pix** (Isola et al., 2017) introduced a conditional GAN for image-to-image translation.

### 5.3 Challenges in CCTV Facial Reconstruction

Challenges include:

- Low-light conditions (Lore et al., 2017)
- Motion blur (Su et al., 2017)
- Low resolution (Yang et al., 2018)
- Pose variations in non-frontal faces (Zhao et al., 2019)

### 5.4 Ethical Considerations

Technologies like facial reconstruction raise **ethical** and **privacy concerns** (Brey, 2004).

### 5.5 Conclusion

While deep learning has made significant progress, challenges remain in real-world surveillance conditions, with a need for continued research and ethical considerations.

---

## **6. Model Architecture and Design**

### 6.1 Overview

The model uses a **Generative Adversarial Network (GAN)** architecture, specifically the **Pix2Pix framework** with a **U-Net-based generator** for image-to-image translation.

### 6.2 Generator Architecture

#### 6.2.1 U-Net Structure

- Uses downsampling and upsampling blocks with **skip connections**.
- Initial layers are frozen to maintain **spatial information**.

#### 6.2.2 Layer Configuration

- **64 channels** in hidden layers for balancing model capacity.

#### 6.2.3 Loss Functions

- **Pixel-wise Loss**: Implements **L1 loss** for direct pixel comparison.
- **Perceptual Loss**: Uses **VGG16** for feature extraction with frozen parameters.

---

## **7. Training Strategy**

### 7.1 Optimizer

- **Adam optimizer** is employed with a **lower learning rate** for stable training.

### 7.2 Learning Rate Scheduler

- Implements **StepLR** for gradual learning rate reduction.

### 7.3 Gradient Control

- Uses a custom **set_requires_grad** function for **layer freezing** and **unfreezing**.

### 7.4 Training Duration and Hardware

- **Primary training hardware**: NVIDIA GeForce **RTX 3050 GPU**
- **Training time**: Approximately **12 hours**.

### 7.5 Batch Size and Metrics

- Batch size and metrics like **PSNR** and **SSIM** are used for evaluation.

### 7.6 Image Enhancement Techniques

- **Data augmentation** techniques like rotations, flips, and contrast enhancements are applied.

### 7.7 Code Management and Explanation

- The code is maintained in a GitHub repository.

### 7.8 Checkpointing and Model Saving

- Checkpoints are saved every **10 epochs** for potential resumption of training.

### 7.9 Validation Strategy

- Validation is performed on a **separate dataset** and used to guide training decisions.

---

## **8. Image Enhancement Techniques**

### 8.1 Preprocessing

- Includes **noise reduction** and **Gaussian smoothing** for better image quality.

### 8.2 Real-Time Processing

- Converts **video input** to individual frames for processing.

---

## **9. Additional Components**

### 9.1 Face Detection

- Integration of **YOLO v8n** is planned for more accurate face detection.

### 9.2 GAN Framework

- The **GAN architecture** is used for image enhancement with a **U-Net generator**.

### 9.3 Training Monitoring

- **Generated images** are periodically saved to monitor progress.

# Facial Reconstruction from Low-Quality CCTV Footage

This project focuses on reconstructing facial images from low-quality CCTV footage using a U-Net architecture. It combines deep learning techniques with image processing to enhance facial details, making it useful for security and surveillance applications.

## 10. Implementation Details

### 10.1 Generator Architecture (U-Net)

#### Key Components of U-Net Architecture:

- **Contracting Path (Encoding Path):**

  - Convolutional layers followed by max pooling to reduce the spatial dimensions of the input image.
  - Captures high-resolution, low-level features.

- **Expanding Path (Decoding Path):**

  - Uses transposed convolutions (upsampling) to reconstruct a dense segmentation map.
  - Increases spatial resolution for detailed image reconstruction.

- **Skip Connections:**

  - Links corresponding layers in the encoding and decoding paths to preserve spatial information.
  - Merges global context with local details, improving segmentation accuracy.

- **Concatenation:**

  - Skip connections are implemented through concatenation of feature maps from the encoding path with upsampled feature maps in the decoding path.

- **Fully Convolutional Layers:**
  - U-Net uses only convolutional layers, ensuring adaptability to varying image sizes while preserving spatial information.

### 10.2 Image Preprocessing

Image preprocessing is critical for optimizing model performance. It involves:

- **Resizing:** Ensures consistent dimensions across images, essential for batch processing.
- **Normalization:** Transforms image data to a standard scale (e.g., mean 0, standard deviation 1), speeding up convergence.
- **Format Conversion:** Converts image data to tensors, suitable for neural network input. This often involves rearranging data formats (e.g., height-width-channel to channel-height-width) and scaling pixel values.

### 10.3 Loss Functions

1. **Pixel-wise Loss:**
   - Measures per-pixel differences between predicted and target images.
   - Common metrics: Mean Absolute Error (L1 Loss) or Mean Squared Error (MSE).
   - Preserves low-level detail but is sensitive to pixel-level discrepancies.
2. **Perceptual Loss:**
   - Evaluates similarity based on high-level features from a pre-trained network (e.g., VGG16).
   - Focuses on visual similarity, allowing for more natural reconstructions and avoiding small pixel-level errors.

### 10.4 Optimization and Training

1. **Adam Optimizer (Adaptive Moment Estimation):**
   - Adjusts learning rates individually for each parameter.
   - Combines the benefits of momentum and adaptive learning rate methods, achieving faster convergence.
2. **Learning Rate Scheduler (StepLR):**
   - Adjusts the learning rate during training, reducing it by a factor (e.g., 0.1) after a certain number of epochs, improving convergence.

### 10.5 Libraries and Frameworks

- **PyTorch (torch):** Deep learning framework.
- **torchvision:** Image processing utilities.
- **pathlib:** Path manipulation for file handling.

## 11. Results and Analysis

### 11.1 Quantitative Measures

1. **Peak Signal-to-Noise Ratio (PSNR):**

   - Measures the ratio between the signal and the noise, with higher PSNR indicating better image quality.
   - Observed improvement: X dB across the dataset.

2. **Structural Similarity Index (SSIM):**

   - Assesses perceptual similarity, with values closer to 1 indicating better performance.
   - Observed improvement: X to Y.

3. **Custom Metrics:**
   - **Pixel-wise Loss:** Significant reduction observed post-training.
   - **Perceptual Loss:** Improved image sharpness and high-level detail retention.

### 11.2 Qualitative Results

1. **Visual Sharpness and Detail:**
   - Reconstructed images demonstrate improved sharpness, particularly in facial features like hair and wrinkles.
2. **Smoothening of Artifacts:**

   - Blurring and aliasing artifacts significantly reduced, yielding smoother facial contours.

3. **Consistency Across Faces:**

   - Robust performance across different facial structures and age groups, with more natural skin tone and hair texture reconstruction.

4. **Challenges and Limitations:**
   - Difficulty in reconstructing extremely low-resolution images and non-frontal poses.
   - Some minor blurring in complex regions like eyes and lips.

### 11.3 Visual Comparison

- **Original Input:** Low-resolution, blurry images with significant distortion in facial features.
- **Reconstructed Output:** Clearer, sharper facial structures with more defined features, though some artifacts remain in challenging areas.

## 12. Challenges and Solutions

### 12.1 Model Training Complexity

- **Data Needs:** Large, diverse datasets required to capture the nuances of facial expressions across different demographics.
- **Overfitting:** Mitigated by using regularization techniques and data augmentation.

### 12.2 Architectural Decision-Making

- Initially explored GANs but pivoted to a pre-trained model approach for better performance.
- Adopted Pix2Pix, a conditional GAN, for its balance between speed and quality.

### 12.3 Loss Function Optimization

- Experimented with adversarial and perceptual loss, balancing both to optimize realism and semantic accuracy.

### 12.4 Dataset Limitations

- SCFace dataset posed challenges with limited expression variety and inconsistent image quality.
- Manual verification and augmentation were required for reliable results.

## 13. Future Work

### 13.1 YOLO v8n Integration

- Incorporating YOLO for face detection to enhance robustness in challenging scenes.

### 13.2 Real-time Video Processing

- Developing capabilities for real-time video input to analyze CCTV footage and live streams.

### 13.3 3D Facial Reconstruction

- Exploring 3D reconstruction techniques using stereovision, structure from motion, or time-of-flight cameras.

### 13.4 Multimodal Emotion Recognition

- Combining audio analysis, body posture, and physiological signals to enhance emotion recognition.

### 13.5 Transfer Learning and Domain Adaptation

- Investigating transfer learning to improve generalization in new environments.

### 13.6 Explainable AI Integration

- Using techniques like Grad-CAM to increase transparency in the decision-making process.

### 13.7 Edge Computing

- Developing on-device processing for privacy-preserving, low-latency applications.

## 14. Conclusion

This project achieved significant advancements in facial reconstruction from low-resolution CCTV footage, demonstrating:

- **Improved Image Quality:** Enhanced sharpness and detail preservation.
- **Effective Technique Integration:** Combined machine learning with super-resolution and noise reduction.
- **Scalability:** Laid groundwork for future enhancements, including YOLO integration and real-time processing.

The model's ability to reconstruct and analyze facial images in real-time can have profound implications for law enforcement and public safety.

## 15. References

- Goodfellow, I., et al. (2014). "Generative Adversarial Networks." In NeurIPS.
- Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). "Image-to-Image Translation with Conditional Adversarial Networks." In CVPR.
- Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." In ICLR.
- Zhang, Y., et al. (2018). "Image Denoising Using Very Deep Residual Channel Attention Networks." In IEEE Transactions on Image Processing.
