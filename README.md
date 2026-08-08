# 📦 Project Title: Digit Classification with Bounding Box Prediction(Text Detection)

## 📖 Overview

This project demonstrates a deep learning approach to **digit classification and localization** using TensorFlow. The model is trained on the MNIST dataset and is designed to both:

* Classify handwritten digits (0–9)
* Predict bounding boxes around the digits within an image

It combines **computer vision** and **multi-output neural networks** to perform classification and regression simultaneously.

---

## 🚀 Features

* 📊 Digit classification using Convolutional Neural Networks (CNN)
* 📦 Bounding box regression for object localization
* 🔄 Data preprocessing and augmentation
* 📈 Training and validation performance visualization
* 📐 Intersection over Union (IoU) evaluation metric
* 🖼️ Visualization of predictions with bounding boxes

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* PIL (Python Imaging Library)
* TensorFlow Datasets (TFDS)

---

## 📂 Project Structure

```
project/
│── main.py / notebook
│── README.md
│── requirements.txt
│
├── data/
├── models/
├── utils/
└── outputs/
```

---

## ⚙️ How It Works

### 1. Data Preparation

* Uses MNIST dataset via TensorFlow Datasets
* Images are padded and resized to 75x75
* Bounding boxes are generated dynamically

### 2. Model Architecture

The model has two outputs:

* **Classification Head** → Predicts digit (0–9)
* **Bounding Box Head** → Predicts (xmin, ymin, xmax, ymax)

### 3. Training

* Loss Functions:

  * Classification → Categorical Crossentropy
  * Bounding Box → Mean Squared Error (MSE)
* Optimizer: Adam
* Trained for multiple epochs with batch processing

### 4. Evaluation

* Accuracy for classification
* MSE for bounding box prediction
* IoU (Intersection over Union) for localization quality

---

## 📊 Results

* Displays predicted vs actual labels
* Visualizes bounding boxes (True vs Predicted)
* Shows training metrics like:

  * Loss
  * Accuracy

---

## ▶️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python main.py
```

---

## 📌 Requirements

* Python 3.x
* TensorFlow
* NumPy
* Matplotlib
* Pillow

---

## 🔍 Future Improvements

* Use more advanced architectures (ResNet, EfficientNet)
* Improve bounding box accuracy
* Add real-world object detection dataset
* Deploy as a web app

---

## 👨‍💻 Author

**Somnath Mohanty**

---
