# 🌱 Plant Seedlings Classification
Distinguishing weeds from crop seedlings with deep learning

## Table of Contents
- [Project Description](#project-description)
- [Key Concepts](#key-concepts)
- [Dataset Exploration](#dataset-exploration)
- [Image Preprocessing](#image-preprocessing)
- [Model Development](#model-development)
  - [Baseline CNN Model](#baseline-cnn-model)
  - [Transfer Learning Approaches](#transfer-learning-approaches)
    - [EfficientNetB0](#efficientnetb0)
    - [VGG19](#vgg19)
    - [Fine-Tuning VGG19](#fine-tuning-vgg19)
- [Zero-Shot Segmentation](#zero-shot-segmentation)
  - [SAM (Segment Anything Model)](#sam-segment-anything-model)
  - [CLIPSeg](#clipseg)
- [Challenges and Obstacles](#challenges-and-obstacles)
- [Conclusions](#conclusions)
- [🚀 Run this Project](#run-this-project)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

---

## Project Description
This project is part of a **Kaggle challenge** aimed at classifying images of young plants (seedlings) into different species. The main goal was to develop a robust **Machine Learning pipeline** using **Computer Vision** techniques to automate the classification process.

You can find the **original Kaggle competition** and download the dataset from the following link:  
[Plant Seedlings Classification - Kaggle](https://www.kaggle.com/c/plant-seedlings-classification)

The project consists of **two phases**:
1. **Standard Image Classification:** Using **raw images** to train a deep learning model.
2. **Zero-Shot Segmentation + Classification:** Applying **Zero-Shot Segmentation** techniques (**SAM & CLIPSeg**) to preprocess the images before training, aiming for improved accuracy.

---

## Key Concepts
Before diving into the development process, here are some fundamental concepts used in this project:

- **Computer Vision**: AI that enables computers to interpret images/videos. Common applications: **facial recognition, object detection**.
- **Convolutional Neural Networks (CNNs)**: Neural networks specialized in image processing.
- **Data Augmentation**: Artificially increasing dataset size (rotation, zoom, flipping, etc.).
- **Class Imbalance**: Some classes have significantly more samples than others → **Used Class Weights** to mitigate bias.
- **Overfitting**: Model memorizes noise instead of patterns → **Used Dropout, Early Stopping, and Data Augmentation**.
- **Transfer Learning**: Reusing **pre-trained models** (**EfficientNet, VGG19, ResNet50**) for new tasks.
- **Fine-Tuning**: Adjusting pre-trained models by **freezing/unfreezing layers** and using a **low learning rate** for effective training.
- **Zero-Shot Learning**: Classifying images **without explicitly training on them**, using **CLIPSeg and SAM** for segmentation.

---

## Dataset Exploration
The dataset consists of:
- **Train Folder**: Labeled images categorized into subfolders by species.
- **Test Folder**: Unlabeled images for predictions.
- **sample_submission.csv**: Template for Kaggle submission.

**Key Findings from Data Exploration:**
- **Class imbalance detected** → Required balancing techniques.
- **Inconsistent image sizes** → **Resized all to 128x128 pixels**.
- **Data Augmentation needed** → Applied rotation, zoom, flipping, and shifts.

---

## Image Preprocessing
- **Resized all images to 128x128 pixels** to optimize GPU memory and speed up training.
- **Applied Data Augmentation**: Rotations, shifts, zoom, and flips.
- **Implemented Class Weights** to balance model learning.

---

## Model Development

### Baseline CNN Model
The first approach was training a **custom CNN from scratch** with:  
- **Four convolutional layers** (32, 64, 128, 256 filters).  
- **Batch Normalization and Dropout** to improve generalization.  
- **L2 regularization** to reduce overfitting.  
- **Categorical Crossentropy loss** for multi-class classification.  
- **Adam optimizer** with an initial learning rate of **0.001**.  

#### Results:
- **Initial Training Accuracy:** **~82%**  
- **Initial Validation Accuracy:** **~86%**  
- **After Fine-Tuning:**  
  - **Training Accuracy:** **~89%**  
  - **Validation Accuracy:** **~90%**  

### Conclusion:
The CNN significantly improved after increasing depth, adding regularization, and applying fine-tuning. However, exploring **transfer learning** with pre-trained models could yield even better results.

---

### Transfer Learning Approaches
To improve performance, I experimented with **pre-trained models**.

#### EfficientNetB0
- Optimized for **accuracy vs. computation trade-off**.
- Fine-tuned the **last 50 layers** while freezing the rest.
- **Did not converge well (~11% accuracy)** → Abandoned.

#### VGG19
- **Deeper network with strong feature extraction**.
- **Trained with additional dense layers & Dropout**.
- **Achieved ~91% accuracy**, making it the best-performing model.

#### Fine-Tuning VGG19
- Unfroze **last 5 layers** and applied **a lower learning rate (1e-6)**.
- **Final accuracy: 91%** on Kaggle.

---

## Zero-Shot Segmentation
Since removing background **could improve classification**, I tested **SAM** and **CLIPSeg**:

### SAM (Segment Anything Model)
- **Automatically generates object masks** without prior training.
- **Fast, but created too many small mask fragments**, making training harder.

### CLIPSeg
- **Zero-shot segmentation using text prompts**.
- Used `"plant"` as a text prompt for isolating only the plant.
- **More refined than SAM, but still had segmentation inconsistencies**.

**Classification Results After Segmentation**:
- **VGG19 on raw images**: **91% accuracy**
- **VGG19 on segmented images**: **86% accuracy**
- **Unexpectedly, segmentation reduced accuracy**, likely due to loss of key features.

---

## Challenges and Obstacles
1. **GPU Memory Issues**
   - Training large models on high-resolution images led to memory errors.
   - **Solution:** Used **Colab Pro+, reduced batch sizes, and froze layers**.

2. **Class Imbalance**
   - Some species had very few images.
   - **Solution:** Applied **Class Weights** and **Data Augmentation**.

3. **Loss of Model Progress**
   - Modifying model structure caused training loss resets.
   - **Solution:** Frequently saved models and avoided structure changes.

4. **Unexpected Segmentation Results**
   - Segmented images **performed worse** than raw images.
   - **Possible reasons:** Loss of key features, imprecise segmentation.

---

## Conclusions
- **VGG19** performed best, achieving **91% accuracy** on Kaggle.
- **Zero-Shot Segmentation unexpectedly reduced accuracy**, but refining segmentation could help.
- **Future improvements**:
  - Better segmentation masks.
  - Training longer with lower learning rates.
  - Additional hyperparameter tuning.

---

## 🚀 Run this Project
### Prerequisites
- **Google Colab (Pro recommended)**
- **TensorFlow 2.x**
- **PyTorch** (for segmentation models)
- `transformers` library for CLIPSeg

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/plant-seedlings-classification.git
cd plant-seedlings-classification

# Install dependencies
pip install -r requirements.txt
