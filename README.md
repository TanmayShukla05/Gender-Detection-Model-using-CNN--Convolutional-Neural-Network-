# 🧑👩 Gender Detection Model using CNN 

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A robust computer vision and deep learning project that identifies gender (Male/Female) from facial images. The model utilizes OpenCV for real-time face detection and a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras for highly accurate classification.

## 🚀 Key Features

* **Automated Face Extraction**: Leverages OpenCV's `haarcascade_frontalface_default.xml` to scan raw images, detect human faces, and automatically crop them to isolate relevant features.
* **Custom CNN Architecture**: A sequential neural network optimized for spatial feature extraction using Convolutional and MaxPooling layers.
* **Live Webcam Integration**: Custom JavaScript integration allowing users to capture images via webcam directly inside Google Colab for real-time predictions.
* **Visual Output**: Draws bounding boxes around detected faces and overlays the predicted gender alongside a confidence percentage.

## 📊 Dataset

The model is trained on the [Gender Dataset by Yasser Hessein](https://www.kaggle.com/datasets/yasserhessein/gender-dataset) downloaded directly via Kaggle API. 
* **Total Processed Images**: 24,576 faces
* **Data Split**: 
  * Training: 16,384 images
  * Validation: 4,096 images
  * Testing: 4,096 images
* **Preprocessing**: Faces are cropped, resized to `64x64` pixels, and normalized (scaled by `/ 255.0`).

## 🧠 Model Architecture

The Sequential CNN model consists of the following layers:
1. **Conv2D** (32 filters, 3x3 kernel, ReLU) + **MaxPooling2D** (2x2)
2. **Conv2D** (64 filters, 3x3 kernel, ReLU) + **MaxPooling2D** (2x2)
3. **Flatten** layer
4. **Dense** (64 units, ReLU)
5. **Dropout** (50% rate to prevent overfitting)
6. **Dense Output** (2 units, Softmax for Male/Female classification)

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Performance:** Reached **93.8% Accuracy** on unseen test data.

## ⚙️ Setup and Usage

### Prerequisites
* Python 3.x
* Jupyter Notebook / Google Colab
* Required Libraries: `tensorflow`, `numpy`, `matplotlib`, `opencv-python`, `kagglehub`

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/TanmayShukla05/Gender-Detection-Model-using-CNN--Convolutional-Neural-Network-.git
