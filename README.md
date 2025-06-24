# 🧠 CIFAR-10 Image Classifier using CNN (TensorFlow/Keras)

This deep learning project builds a powerful **Convolutional Neural Network (CNN)** that classifies images from the **CIFAR-10 dataset** into 10 different categories. The model is trained using **data augmentation** for better generalization and tested on both the test set and a custom image (`cat.jpg`).

---

## 📁 Dataset

- **Source**: `tensorflow.keras.datasets.cifar10`
- **Classes**:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`,  
    `dog`, `frog`, `horse`, `ship`, `truck`
- **Image size**: 32x32 pixels
- **Total images**:
  - Training: 50,000  
  - Testing: 10,000

---

## 🏗️ Model Architecture

```text
Input Image → Conv2D (32) → MaxPooling  
            → Conv2D (64) → MaxPooling  
            → Conv2D (32) → MaxPooling  
            → Conv2D (16)  
            → Flatten  
            → Dense (64) + Dropout (0.5)  
            → Dense (10, softmax)
