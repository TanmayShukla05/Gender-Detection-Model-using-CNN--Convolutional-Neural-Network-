# 🧑👩 Real-Time Gender Detection Model using CNN 

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A robust computer vision and deep learning project that identifies gender (Male/Female) from facial images. The standout feature of this project is a **Live Camera Integration** that allows users to use their webcam for real-time gender prediction directly inside a Google Colab environment.

## 🚀 Key Features

* **🎥 Live Camera / Webcam Prediction**: Features custom JavaScript code that activates the user's webcam, captures a live photo, and instantly feeds it into the model for real-time gender classification and confidence scoring.
* **🤖 Custom CNN Architecture**: A sequential neural network built from scratch using TensorFlow/Keras, optimized for spatial feature extraction using Convolutional, MaxPooling, and Dropout layers.
* **🖼️ Automated Face Extraction**: Leverages OpenCV's `haarcascade_frontalface_default.xml` to automatically scan images (both static and live webcam feeds), detect human faces, and crop them out.
* **📈 Visual Bounding Boxes**: Automatically draws green bounding boxes around detected faces and overlays text displaying the predicted gender alongside a confidence percentage (e.g., `Male - 98.50%`).

## 📊 Dataset & Preprocessing

The model is trained on the [Gender Dataset by Yasser Hessein](https://www.kaggle.com/datasets/yasserhessein/gender-dataset) downloaded directly via the Kaggle API. 
* **Total Processed Images**: 24,576 faces
* **Data Split**: 
  * Training: 16,384 images
  * Validation: 4,096 images
  * Testing: 4,096 images
* **Preprocessing Pipeline**: Faces are detected, cropped, resized to `64x64` pixels, and normalized (scaled by `/ 255.0`).

## 🧠 Model Architecture & Performance

The Sequential CNN model consists of the following layers:
1. **Conv2D** (32 filters, 3x3) + **MaxPooling2D** (2x2)
2. **Conv2D** (64 filters, 3x3) + **MaxPooling2D** (2x2)
3. **Flatten** layer
4. **Dense** (64 units, ReLU)
5. **Dropout** (50% rate to prevent overfitting)
6. **Dense Output** (2 units, Softmax)

**Performance:** The model achieved **93.8% Accuracy** on the unseen test dataset.

## ⚙️ Setup and Usage

### Prerequisites
* Python 3.x
* Jupyter Notebook / Google Colab (Recommended for webcam feature)
* Required Libraries: `tensorflow`, `numpy`, `matplotlib`, `opencv-python`, `kagglehub`

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/TanmayShukla05/Gender-Detection-Model-using-CNN--Convolutional-Neural-Network-.git
