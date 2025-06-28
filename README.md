# 🐶🐱 Cat vs Dog Image Classifier

This project is a deep learning-based image classification system that can automatically identify whether a given image contains a **cat** or a **dog**. It uses **TensorFlow** and **Keras** with **transfer learning** (MobileNetV2) for high performance and is deployed using a **Streamlit** web interface.

---

## 📸 Demo

![Classifier Demo](https://i.imgur.com/9j0Gq3H.gif)

---

## 📦 Features

- ✅ Image classification using Convolutional Neural Networks (CNNs)
- ✅ Transfer learning with **MobileNetV2**
- ✅ Real-time prediction via a **Streamlit** web app
- ✅ Data augmentation for better generalization
- ✅ Easily expandable with new images

---

## 🛠️ Technologies Used

- Python 3.11+
- TensorFlow & Keras
- MobileNetV2 (pre-trained model)
- Streamlit (web interface)
- Pillow, NumPy, Matplotlib
- OpenCV (optional for extra processing)

---

1.## 📁 Project Structure

├── dataset/
│ ├── train/
│ │ ├── cats/
│ │ └── dogs/
│ └── validation/
│ ├── cats/
│ └── dogs/
├── image_classifier.py # Training script
├── app.py # Streamlit web app
├──cat_dog_classifier.h5 # Trained model

2. ***Install Requirements***

pip install tensorflow streamlit pillow numpy matplotlib

3.***Prepare the Dataset***
Organize images as follows:
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── validation/
    ├── cats/
    └── dogs/
4.***Train the Model***
python image_classifier.py
5.***Run the Web App***
streamlit run app.py
Open in your browser: http://localhost:8501

👨‍💻 Author
Umer Murthala Thangal K K
B.Tech Computer Science Student | AI Enthusiast | Final Year


