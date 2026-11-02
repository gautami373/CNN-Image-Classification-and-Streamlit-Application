# CNN-Image-Classification-and-Streamlit-Application
CNN Image Classification and Streamlit Application

# CNN Handwritten Digit Recognition

A handwritten digit recognition system built using a Convolutional Neural Network (CNN), TensorFlow, and Streamlit.

The project uses the MNIST dataset to classify handwritten digits from 0 to 9. It also provides a web interface where users can upload an image of a handwritten digit and receive a prediction.

## Project Overview

Handwritten digit recognition is an image classification problem in which a machine learning model identifies numerical digits from images.

This project demonstrates:
- Loading and preprocessing image data
- Building a CNN model using TensorFlow and Keras
- Training and validating the model
- Evaluating model performance on test data
- Displaying actual and predicted digit labels
- Creating a Streamlit application for image prediction

## Objectives

- Understand the fundamentals of CNN-based image classification.
- Train a model using the MNIST dataset.
- Evaluate the model using test data.
- Build a simple, interactive web application.
- Display the predicted digit and estimated confidence.

## Dataset

**Dataset:** MNIST Handwritten Digits

The MNIST dataset contains grayscale images of handwritten digits.

| Property | Description |
|---|---|
| Training images | 60,000 |
| Testing images | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image type | Grayscale |
| Number of classes | 10 |
| Classes | Digits 0–9 |

The dataset is loaded automatically using TensorFlow/Keras.

## Technologies Used

- **Python** — programming language
- **TensorFlow and Keras** — CNN model development and training
- **NumPy** — numerical operations and image preprocessing
- **Matplotlib** — training graphs and prediction visualizations
- **Pillow (PIL)** — image loading and processing
- **Streamlit** — interactive web application

## CNN Model Architecture

The CNN model consists of the following layers:

1. Input layer: 28 × 28 grayscale image
2. Convolutional layer: 32 filters with ReLU activation
3. Max pooling layer
4. Convolutional layer: 64 filters with ReLU activation
5. Max pooling layer
6. Flatten layer
7. Dense layer: 128 neurons with ReLU activation
8. Output layer: 10 neurons with Softmax activation

### Model Configuration

- Optimizer: Adam
- Loss function: Sparse Categorical Cross-Entropy
- Evaluation metric: Accuracy
- Epochs: 5
- Batch size: 64
- Validation split: 10%

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
