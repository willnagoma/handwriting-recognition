# Handwritten Character Recognition using CNN

This project demonstrates how to build a convolutional neural network (CNN) using TensorFlow and Keras to recognize handwritten English characters (A–Z) from grayscale images.

It was designed as a personal learning project to deepen my understanding of image classification, data preprocessing, and model deployment using Streamlit. The model is trained on a publicly available dataset containing over 370,000 labeled characters and achieves strong performance using a relatively simple CNN architecture.

---

## Key Skills Demonstrated

- Applied deep learning techniques using TensorFlow and Keras to build a CNN for image classification.
- Preprocessed large-scale image data using NumPy, Pandas, and OpenCV.
- Implemented one-hot encoding, reshaping, and normalization to prepare data for training.
- Designed and trained a multi-layer convolutional neural network with dropout and ReLU activations.
- Evaluated model performance using validation accuracy and confusion matrix visualization.
- Saved and loaded trained models for real-time inference.
- Developed a Streamlit web application to interactively test model predictions with uploaded images.

---

## Project Overview

- **Goal**: Classify handwritten characters (A–Z) based on 28x28 grayscale pixel data.
- **Framework**: TensorFlow + Keras
- **Model Type**: Convolutional Neural Network (CNN)
- **Frontend**: Streamlit app for real-time predictions on uploaded handwriting samples.
- **Dataset**: [A–Z Handwritten Data CSV](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

---

## Project Structure

```bash
handwriting-recognition/
├── data/
│   └── handwriting_data.csv               # Dataset
├── model/
│   └── handwritten_character_recog_model.h5  # Trained model
├── logs/                                  # Optional training logs
├── src/
│   ├── train.py                           # Model training script
│   └── preprocess.py                      # Image preprocessing utilities
├── app/
│   └── streamlit_app.py                   # Frontend application (in progress)
├── requirements.txt
└── README.md
