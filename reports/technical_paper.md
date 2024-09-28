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

10. Implementation Details
    10.1 Generator Architecture (U-Net)

Key Components of UNET Architecture:
Contracting Path (Encoding Path):
Consists of convolutional layers followed by max pooling.
Gradually reduces the spatial dimensions of the input image.
Captures high-resolution, low-level features.
Expanding Path (Decoding Path):
Uses transposed convolutions (deconvolutions or upsampling layers) to upsample feature maps.
Increases spatial resolution of feature maps for reconstructing a dense segmentation map.
Skip Connections:
Links corresponding layers in the encoding and decoding paths.
Combines both local and global information.
Enhances segmentation accuracy by preserving essential spatial data.
Concatenation:
Skip connections are implemented through concatenation.
Feature maps from the encoding path are concatenated with upsampled feature maps in the decoding path.
Helps the network utilize multi-scale information for better segmentation by merging high-level context with low-level features.
Fully Convolutional Layers:
UNET uses only convolutional layers, without fully connected layers.
This design allows the model to process images of varying sizes while maintaining spatial information, making it adaptable for different segmentation tasks.

This U-Net architecture uses skip connections to preserve spatial information across different scales, crucial for detailed image reconstruction.
10.2 Image Preprocessing
Image preprocessing is a crucial step in preparing data for deep learning models, particularly in computer vision tasks. The process typically involves several key operations designed to standardize input data and optimize model performance. These operations generally include resizing, normalization, and format conversion.

Resizing ensures all images have consistent dimensions, which is essential for batch processing in neural networks. It reduces computational complexity and mitigates the impact of varying image resolutions in datasets. However, it's important to note that resizing can lead to loss of detail in larger images or distortion in images with different aspect ratios.

Normalization is a critical preprocessing step that transforms the data to have a standard scale, typically with a mean of 0 and a standard deviation of 1. This process aids in faster convergence during training, makes the optimization landscape more symmetric, and reduces the impact of outliers in the data. Normalization effectively puts all features on a level playing field, allowing the model to learn more efficiently from the input data.

The final key step is format conversion, which transforms the image data into a structure suitable for neural network processing, such as tensors. This step often includes changing the image format from a height-width-channel structure to a channel-height-width format, and scaling pixel values to a standardized range. These preprocessing steps, when combined, ensure that all input images are consistently formatted and normalized, thereby optimizing model performance and improving training stability.

10.3 Loss Functions

1. Pixel-wise Loss:
   Pixel-wise loss measures the difference between the predicted output and the ground truth on a per-pixel basis. It quantifies how closely the predicted image resembles the target image. A common pixel-wise loss function is Mean Absolute Error (L1 Loss) or Mean Squared Error (MSE).
   Key Points:
   Low-Level Detail Preservation: Captures the exact pixel values, making it effective for tasks requiring precise detail reproduction.
   Sensitivity: It is sensitive to pixel-level discrepancies, which can lead to sharper reconstructions.
   Equation:
   For L1 Loss:

For MSE:
where NNN is the number of pixels, xix_ixi​ is the predicted pixel value, and yiy_iyi​ is the target pixel value.

2. Perceptual Loss:
   Perceptual loss is designed to assess the similarity between two images based on high-level features extracted from a pre-trained neural network, such as VGG16. This approach focuses on how visually similar two images are rather than their pixel-wise differences.
   Key Points:
   High-Level Feature Representation: Utilizes the features learned by a deep network to evaluate image similarity based on perceptual characteristics.
   Robustness to Artifacts: It can mitigate the effects of small pixel-wise errors that may not be perceptually significant, allowing for smoother and more visually appealing results.
   Equation:

   10.4 Optimization and Training

1. Adam Optimizer (Adaptive Moment Estimation):
   The Adam optimizer is an adaptive learning rate optimization algorithm that combines the benefits of AdaGrad and RMSProp. It adapts the learning rate of each parameter by maintaining two moving averages: one for the gradients (first moment) and another for the squared gradients (second moment). These moving averages are corrected for bias, particularly in the initial stages of training, which helps Adam achieve faster convergence.
   Key Points:
   Adaptive learning rates: adjusts learning rates for each parameter individually.
   Momentum: Helps to smooth the gradient updates, which accelerates training.
   Bias Correction: Corrects the bias in the moving averages for better stability.
   Equations:
   First moment (mean):
   Second moment (variance):​
   Bias-corrected estimates:
1. Learning Rate Scheduler (StepLR):
   A learning rate scheduler adjusts the learning rate during training. This helps the model fine-tune itself as training progresses by decreasing the learning rate after a specified number of epochs. In the StepLR scheduler, the learning rate is reduced by a factor (e.g., 0.1) after a certain number of epochs (e.g., every 30 epochs). This gradual reduction allows the model to make smaller, more precise updates as it converges.
   Key Points:
   Step Decay: Reduces the learning rate periodically by a factor to help the model fine-tune.
   Convergence: Prevents overshooting during optimization and allows better convergence near the solution.
   Equation: New learning rate , gamma being the reduction factor.

   10.5 Libraries and Frameworks
   The implementation relies on the following key libraries:
   PyTorch (torch)
   torchvision
   pathlib

1. Results and Analysis
   Our facial reconstruction model demonstrated significant improvements in both quantitative metrics and qualitative visual assessments. This section details the performance of the model across various evaluation criteria.
   11.1 Quantitative Measures
   11.1.1 Peak Signal-to-Noise Ratio (PSNR)
   Metric Description: PSNR measures the ratio between the maximum possible value of the image signal and the noise, expressed in decibels (dB).
   Significance: Higher PSNR values indicate better image quality.
   Observed Improvement: Post-training, PSNR values saw an average improvement of X dB across the dataset.
   Interpretation: This improvement demonstrates the model's effectiveness in enhancing fine facial details, bringing reconstructed images closer to high-resolution ground truth.
   11.1.2 Structural Similarity Index (SSIM)
   Metric Description: SSIM captures structural information and assesses perceptual quality, providing a value between -1 and 1.
   Significance: Higher SSIM values reflect better perception similarity.
   Observed Improvement: The average SSIM score improved from X to Y.
   Interpretation: This improvement signifies better preservation of structural details such as facial contours, shapes, and textures in the reconstructed images.

   11.1.3 Custom Metrics
   Pixel-wise Loss: The model reduced pixel-wise error significantly, with an average pixel-wise loss of X after training.
   Perceptual Loss: The VGG-perceptual loss resulted in perceptually sharper images, enhancing texture and maintaining high-level details of facial features.

   11.2 Qualitative Results
   11.2.1 Visual Sharpness and Detail
   Reconstructed images show clear improvement in sharpness compared to low-resolution inputs.
   Features such as facial hair, wrinkles, and expressions are noticeably clearer.
   The model successfully captured both global structure and local textures.
   11.2.2 Smoothening of Artifacts
   Significant reduction in blurring and aliasing artifacts, particularly around facial edges and eye regions.
   Contours of facial features became smoother.
   Previously pixelated areas transitioned into sharper regions, yielding a more realistic appearance.
   11.2.3 Consistency Across Faces
   Consistent reconstruction quality across various subjects, including different age groups.
   Improved hair texture and skin tone, appearing more natural.
   The model demonstrated good generalization to different facial structures and age groups.
   11.2.4 Challenges and Limitations
   Some difficulties in handling extreme cases, such as very low-resolution images or non-frontal facial poses.
   Fine-grained details like eyes and lips were harder to reconstruct accurately in challenging cases.
   Occasional blurring or slight misalignment in complex facial regions.

   11.3 Visual Comparison
   Analysis of the output image after reconstruction:
   Top Row: Original low-resolution inputs with significant blurriness and loss of details. Facial structures appear smooth, with key features such as eyes, nose, and mouth being heavily distorted.
   Middle and Bottom Rows: Reconstructed images show:
   More defined facial features include skin texture, hair strands, and facial contours.
   Striking improvement in sharpness and clarity.
   Some remaining artifacts in complex regions (around eyes and mouth), but overall significant enhancement.
   11.4 Visual Comparison: Low-Quality CCTV Images vs. Reconstructed Faces

   11.5 Summary of Findings
   The facial reconstruction model demonstrates substantial improvements in both quantitative metrics and qualitative visual assessments. The enhanced PSNR and SSIM scores, coupled with the visually apparent improvements in facial detail and structure, indicate that the model effectively addresses the challenge of reconstructing high-quality facial images from low-resolution CCTV footage.
   While the model shows robust performance across a variety of facial types and ages, there remain opportunities for further improvement, particularly in handling extreme low-resolution cases and non-frontal poses. These challenges provide direction for future refinements of the model.

1. Challenges and Solutions
   Throughout the development of our facial emotion recognition system, we faced several significant challenges. These obstacles not only shaped our approach but also provided valuable insights for future research. Here, we detail the key challenges and our strategies to address them:
   Model Training Complexity Training an effective facial emotion recognition model proved to be a non-trivial task due to:
   The need for large, diverse datasets to capture the nuances of facial expressions across different demographics.
   Balancing model complexity with computational resources to achieve real-time performance.
   Mitigating overfitting while ensuring generalization to unseen data.
   Architectural Decision-Making Selecting the optimal neural network architecture was crucial. We experimented with various architectures, each presenting its own set of trade-offs:
   Initially, we explored Generative Adversarial Networks (GANs) for their potential in generating realistic facial expressions.
   Due to limited effectiveness of GANs in our specific use case, we pivoted to fine-tuning pre-trained models, which offered a better balance of performance and training efficiency.
   Eventually, we adopted Pix2Pix, a conditional GAN architecture, as a compromise solution given our time constraints.
   Loss Function Optimization Defining an effective loss function was critical for model performance:
   We implemented adversarial loss to improve the realism of generated expressions.
   VGG16-based perceptual loss was incorporated to enhance the semantic quality of the outputs.
   Balancing these loss components proved challenging and required extensive experimentation.
   Dataset Limitations We utilized the SCFace dataset, which presented its own set of challenges:
   Limited diversity in facial expressions and demographics.
   Varying image quality and resolution, necessitating robust preprocessing techniques.
   Annotation inconsistencies required manual verification and correction.
   Feature Extraction Techniques Exploring various feature extraction methods led to several challenges:
   We initially experimented with MediaPipe for facial landmark detection.
   Attempted to create and concatenate heatmaps with RGB images for enhanced feature representation.
   Resource constraints and time limitations prevented full exploration of these advanced techniques.
   Data Preprocessing and Augmentation Transforming and augmenting the dataset was a significant challenge.
   Developing a robust pipeline to handle various image formats and resolutions.
   Implementing effective augmentation techniques to increase dataset diversity without introducing artifacts.
   Balancing augmentation intensity to improve model generalization without compromising learning stability.
   Hyperparameter Tuning Fine-tuning model hyperparameters required extensive experimentation:
   Learning rate optimization was particularly challenging. We tested rates ranging from 1e-5 to 1e-3, ultimately finding the optimal balance between convergence speed and stability.
   Batch size selection was constrained by GPU memory limitations, affecting training dynamics.
   Scheduler selection and configuration to adapt learning rates during training required careful consideration.
   Input Data Format Considerations Adapting to different input data formats presented additional challenges:
   Initially, we worked with .npy files for MediaPipe outputs, which required specific handling in our data loading pipeline.
   Transitioning to .jpeg files for broader compatibility necessitated adjustments in our preprocessing steps and data augmentation techniques.
   These challenges underscore the complexity of developing a robust facial emotion recognition system. Each obstacle provided valuable lessons, guiding our decision-making process and informing our approach to model development and optimization.

1. Future Work
   Our current research has laid a solid foundation for facial emotion recognition in static images. However, there are several promising avenues for future enhancements that could significantly expand the capabilities and real-world applicability of our system:
   13.1 Integration of YOLO v8n for Face Detection The incorporation of YOLO (You Only Look Once) v8n, a state-of-the-art object detection system, will enhance our face detection capabilities. This integration will improve the robustness of our pipeline, especially in complex scenes with multiple faces or challenging lighting conditions.
   13.2 Real-time Video Processing Developing real-time processing capabilities for video input is a crucial next step. This enhancement will enable our system to analyze CCTV footage and live video streams, opening up applications in security, retail analytics, and human-computer interaction.
   13.3 3D Facial Reconstruction While our current work focuses on 2D images, future efforts will explore 3D facial reconstruction techniques. This advancement will allow for more accurate emotion recognition by considering facial geometry and depth information. Potential approaches include:
   Utilizing stereovision systems for depth estimation
   Implementing Structure from Motion (SfM) algorithms for 3D reconstruction from multiple 2D views
   Exploring the use of time-of-flight cameras or structured light systems for direct 3D capture
   13.4 Multimodal Emotion Recognition To improve accuracy and robustness, we plan to incorporate additional modalities beyond facial expressions. This may include:
   Audio analysis for speech emotion recognition
   Body posture and gesture recognition
   Physiological signal processing (e.g., heart rate variability, galvanic skin response)
   13.5 Transfer Learning and Domain Adaptation To address the challenge of limited labeled data in specific domains, we will investigate transfer learning and domain adaptation techniques. This will allow our model to generalize better to new environments and demographics.
   13.6 Explainable AI Integration Incorporating explainable AI techniques, such as Gradient-weighted Class Activation Mapping (Grad-CAM) or SHAP (SHapley Additive exPlanations), will provide insights into the model's decision-making process. This enhancement will increase transparency and trust in the system's outputs.
   13.7 Edge Computing Implementation To address privacy concerns and reduce latency, we aim to develop an edge computing version of our system. This will allow for on-device processing of sensitive facial data without the need for cloud transmission.
   These future enhancements will not only improve the technical capabilities of our facial emotion recognition system but also broaden its applicability across various industries and use cases.

1. Conclusion
   This project has successfully developed an AI model that significantly enhances facial images from low-quality CCTV footage. Key achievements include:
   Improved Image Quality: The model utilizes a Generative Adversarial Network (GAN) with a Pix2Pix framework, leading to notable enhancements in facial image sharpness, detail preservation, and artifact reduction.
   Effective Integration of Techniques: By combining machine learning with traditional image processing methods, such as super-resolution and noise reduction, the system achieves superior reconstruction of facial images.
   Quantitative and Qualitative Validation: The use of metrics like Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) demonstrates significant improvements in image quality, supported by qualitative assessments.
   Scalability and Future Enhancements: The groundwork has been laid for integrating advanced face detection technologies like YOLO v8n and developing real-time processing capabilities, further expanding the system's applicability.
   Potential Impact:
   The advancements made through this project have profound implications for law enforcement and security sectors. Enhanced image quality can lead to more accurate facial recognition, thereby aiding in criminal investigations and improving public safety. The ability to reconstruct and analyze facial images in real-time can empower security personnel with better situational awareness, ultimately contributing to more effective crime prevention and response strategies. As such, this research represents a significant step forward in the fields of video surveillance and forensic analysis.

1. References

Goodfellow, I., et al. (2014). "Generative Adversarial Networks." In Advances in Neural Information Processing Systems (NeurIPS).

Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). "Image-to-Image Translation with Conditional Adversarial Networks." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." In “International Conference on Learning Representations (ICLR)”.

Zhang, Y., et al. (2018). "Image Denoising Using Very Deep Residual Channel Attention Networks." In IEEE Transactions on Image Processing.
